"""
RL Trainer

Trains the RL agent for scene reconstruction using Stable-Baselines3.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.rl.environment import SceneReconstructionEnv


logger = get_logger("rl_trainer")


class RLTrainer:
    """Train RL agent for scene reconstruction."""

    def __init__(self, config=None):
        """
        Initialize RL trainer.

        Args:
            config: Configuration object
        """
        self.config = config or load_config()

        # Get RL settings
        self.algorithm = self.config.get("rl.algorithm", "PPO")
        self.learning_rate = self.config.get("rl.learning_rate", 0.0003)
        self.total_timesteps = self.config.get("rl.total_timesteps", 1000000)
        self.n_envs = self.config.get("rl.n_envs", 4)
        self.n_steps = self.config.get("rl.n_steps", 2048)
        self.batch_size = self.config.get("rl.batch_size", 64)

        # Paths
        self.checkpoints_dir = Path(
            self.config.get("paths.checkpoints", "/data/checkpoints")
        )
        self.logs_dir = Path(self.config.get("paths.logs", "/data/logs"))
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.env = None

        logger.info("RL trainer initialized")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Total timesteps: {self.total_timesteps}")

    def create_env(self):
        """Create vectorized training environment."""
        def make_env():
            return SceneReconstructionEnv(config=self.config)

        if self.n_envs > 1:
            self.env = SubprocVecEnv([make_env for _ in range(self.n_envs)])
        else:
            self.env = DummyVecEnv([make_env])

        logger.info(f"Created {self.n_envs} training environments")

    def create_model(self):
        """Create RL model."""
        if self.env is None:
            self.create_env()

        # Select algorithm
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                verbose=1,
                tensorboard_log=str(self.logs_dir / "tensorboard"),
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                verbose=1,
                tensorboard_log=str(self.logs_dir / "tensorboard"),
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                verbose=1,
                tensorboard_log=str(self.logs_dir / "tensorboard"),
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        logger.info(f"Created {self.algorithm} model")

    def train(self):
        """Train the RL agent."""
        if self.model is None:
            self.create_model()

        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get("training.checkpoint.frequency", 10000),
            save_path=str(self.checkpoints_dir),
            name_prefix=f"{self.algorithm}_scene_reconstruction",
        )

        # Train
        logger.info(f"Starting training for {self.total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=self.algorithm,
        )

        # Save final model
        final_model_path = self.checkpoints_dir / f"{self.algorithm}_final.zip"
        self.model.save(str(final_model_path))
        logger.info(f"Training complete. Model saved to {final_model_path}")

    def evaluate(self, n_episodes: int = 10):
        """
        Evaluate trained model.

        Args:
            n_episodes: Number of episodes to evaluate
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        logger.info(f"Evaluating model for {n_episodes} episodes...")

        episode_rewards = []
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0] if isinstance(reward, list) else reward

            episode_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        logger.info(f"Mean reward: {mean_reward:.2f}")

        return episode_rewards


def main():
    """Command-line interface for RL trainer."""
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["PPO", "A2C", "SAC"],
        help="RL algorithm (default: from config)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: from config)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate existing model instead of training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model for evaluation",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    if args.algorithm:
        config._config.setdefault("rl", {})["algorithm"] = args.algorithm
    if args.timesteps:
        config._config.setdefault("rl", {})["total_timesteps"] = args.timesteps

    # Create trainer
    trainer = RLTrainer(config=config)

    if args.eval:
        if args.model_path:
            # Load model
            trainer.create_env()
            if trainer.algorithm == "PPO":
                trainer.model = PPO.load(args.model_path, env=trainer.env)
            elif trainer.algorithm == "A2C":
                trainer.model = A2C.load(args.model_path, env=trainer.env)
            elif trainer.algorithm == "SAC":
                trainer.model = SAC.load(args.model_path, env=trainer.env)

        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
