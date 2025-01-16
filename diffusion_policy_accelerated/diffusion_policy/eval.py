import collections 
import os

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch 
import torch.nn as nn
import gdown 
import numpy as np
from tqdm import tqdm 
import random
import diffusion_policy_accelerated.config as config 
from diffusion_policy_accelerated.diffusion_policy.push_t_env import PushTImageEnv
from diffusion_policy_accelerated.diffusion_policy.model import get_resnet, replace_bn_with_gn, ConditionalUnet1D, load_noise_pred_net_graph, normalize_data, unnormalize_data
import time
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class MDPEnvironment:
    def __init__(self):
        '''
        A class representing the MDP environment for evaluation. 
        Handles keep track of observations, rewards, and changing env states with actions. 
        '''
        self.env = PushTImageEnv()
        self.reset() 

    def reset(self):
        self.env.seed(config.SEED)
        obs, info = self.env.reset()
        self.obs_deque = collections.deque([obs] * config.OBS_HORIZON, maxlen=config.OBS_HORIZON)
        self.rewards = []
        return obs, info

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.obs_deque.append(obs)
        self.rewards.append(reward)
        return obs, reward, done, info, max(self.rewards)

class ModelManager:
    '''
    A class that manages the loading and inference of the diffusion policy model. Handles model initialization, weight loading, 
    CUDA graph generation (in accelerated mode) and predicting actions using diffusion. 
    '''
    def __init__(self):
        self.generator = torch.Generator().manual_seed(config.SEED)
        self.generator_state = self.generator.get_state()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.NUM_DIFFUSION_ITERS,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'

        )
        
        vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(vision_encoder)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config.ACTION_DIM,
            global_cond_dim=config.OBS_DIM*config.OBS_HORIZON
        )

        ckpt_path = os.path.join(os.path.dirname(__file__), "pusht_vision_100ep.ckpt")
        if not os.path.isfile(ckpt_path):
            id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
            gdown.download(id=id, output=ckpt_path, quiet=False)
        state_dict = torch.load(ckpt_path, map_location='cuda')

        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        })
        self.nets.load_state_dict(state_dict)
        self.nets.to(device=config.DEVICE)
        
        # with config.inference_mode_context(config.InferenceMode.ACCELERATED):
        self.noise_pred_net_graph, self.static_noisy_action, self.static_k, self.static_obs_cond, \
        self.static_diffusion_noise, self.static_model_output = load_noise_pred_net_graph(self.noise_pred_net)
    def reset_generator(self):
        """Make sure every time rng is used, noise is identical to initial state."""
        self.generator.set_state(self.generator_state)
        return self.generator
    def predict_action(self, obs_deque):
        """
        Predicts the next action to take based on the current observation deque.

        This method processes the observation deque to extract image and agent position features, normalizes these features, and then
        uses the vision encoder to generate image embeddings. A random noisy action is then generated and denoised through either
        a normal or accelerated inference process depending on the configured inference mode. The final action is unnormalized before
        being returned.

        Parameters:
        - obs_deque (collections.deque): A deque containing the most recent observations.

        Returns:
        - numpy.ndarray: The predicted action to take.
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


        images = np.stack([x['image'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
        nagent_poses = normalize_data(agent_poses, stats=config.STATS['agent_pos'])

        nimages = images
        nimages = torch.from_numpy(nimages).to(config.DEVICE)
        nagent_poses = torch.from_numpy(nagent_poses).to(config.DEVICE)
        # with self.noise_pred_net.eval():
        denoising_steps = 16
        with torch.no_grad():
            image_features = self.vision_encoder(nimages)
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)
            # obs_cond = obs_features.unsqueeze(0).to(config.DEVICE).float().view(1, -1)
            obs_cond = torch.ones((1, 1028), device=config.DEVICE)
            # import pdb; pdb.set_trace()
            # noisy_action = torch.randn((1, config.PRED_HORIZON, config.ACTION_DIM), device=config.DEVICE)
            noisy_action = torch.zeros((1, config.PRED_HORIZON, config.ACTION_DIM), device=config.DEVICE)
            og_noisy_action = noisy_action.clone()
            # if config.INFERENCE_MODE == config.InferenceMode.NORMAL:
            intermediate_eager = []
            # x = noisy_action.clone()
            start_time_eager = time.time()
            for k in self.noise_scheduler.timesteps[:denoising_steps]:
                k = torch.tensor(k, device=config.DEVICE)
                noise_pred = self.noise_pred_net(
                    sample=noisy_action,
                    timestep=k,
                    global_cond=obs_cond
                )
                intermediate_eager.append(noise_pred.detach().cpu().clone())
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action,
                    generator=self.reset_generator()
                ).prev_sample
            end_time_eager = time.time()
            print(f"Time taken for eager execution: {end_time_eager - start_time_eager} seconds")

            obs_cond = torch.ones((1, 1028), device=config.DEVICE)
            
            self.static_model_output.copy_(og_noisy_action)
            intermediate_graph = []

        start_time = time.time()
        for k in self.noise_scheduler.timesteps[:denoising_steps]:
            self.static_k.copy_(k)
            self.static_obs_cond.copy_(obs_cond)
            self.static_noisy_action.copy_(self.static_model_output) 
                # <-- The "sample" from previous iteration

            # 1) Replay the graph => new noise_pred is stored in self.static_model_output
            self.noise_pred_net_graph.replay()
            assert not torch.equal(self.static_model_output, og_noisy_action)
            # 2) Use the fresh model output from self.static_model_output, not the old "noise_pred"
            noise_pred = self.static_model_output.clone()
            intermediate_graph.append(noise_pred.detach().cpu().clone())
            # 3) Then do the scheduler step exactly as in normal pass
            next_sample = self.noise_scheduler.step(
                model_output=noise_pred, 
                timestep=k, 
                sample=self.static_noisy_action,
                generator=self.reset_generator()
            ).prev_sample

            # 4) Overwrite self.static_model_output with the newly updated sample
            self.static_model_output.copy_(next_sample)

            # assert noisy_action.shape == self.static_model_output.shape
            # assert torch.allclose(noisy_action, self.static_model_output, atol=1e-3)

        end_time = time.time()
        print(f"Time taken for the function: {end_time - start_time} seconds")
            
        # for i, (e, g) in enumerate(zip(intermediate_eager, intermediate_graph)):
        #     diff = (e - g).abs()
        #     print(f"Step {i}: max diff={diff.max()}, mse={diff.pow(2).mean()}")

        # import pdb; pdb.set_trace()



            ## Check if the noisy action is the same as the static model output or at least very close

        diff = torch.abs(noisy_action[0] - self.static_model_output[0])
        max_diff = torch.max(diff).item()
        min_diff = torch.min(diff).item()
        mse_diff = torch.mean(diff ** 2).item()
        worst_offenders = torch.argmax(diff, dim=-1)
        print(f"Max difference: {max_diff}, Min difference: {min_diff}, MSE difference: {mse_diff}")
        # print(f"Worst offending dimensions: {worst_offenders}")
        import pdb; pdb.set_trace()

        naction = noisy_action.detach().to('cpu').numpy()
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=config.STATS['action'])
        return action_pred

def run_evaluation(env_handler, model_manager):
    '''
    Evaluates the model by resetting the environment, executing predicted actions, and collecting rewards until 
    the episode ends or reaches the maximum steps, returning the highest reward.

    Parameters:
    - env_handler (MDPEnvironment): The environment handler to interact with the simulation environment.
    - model_manager (ModelManager): The model manager that predicts actions based on observations.

    Returns:
    - max_rewards (float): The maximum reward achieved during the episode.
    '''
    _, _ = env_handler.reset()
    done = False
    step_idx = 0

    with tqdm(total=config.MAX_STEPS, desc="Eval PushTImageEnv") as pbar:
        while not done and step_idx < config.MAX_STEPS:
            action_pred = model_manager.predict_action(env_handler.obs_deque)
            start = config.OBS_HORIZON - 1
            end = start + config.ACTION_HORIZON
            action = action_pred[start:end,:]

            for i in range(len(action)):
                _, reward, done, _, max_rewards = env_handler.step(action[i])
                
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > config.MAX_STEPS:
                    done = True
                if done:
                    break

    return max_rewards



