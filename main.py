import torch
import argparse
from train import train_agent

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4")
parser.add_argument("--frame_stack", type=int, default=4)
parser.add_argument("--capacity", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--num_frames_to_train", type=int, default=1500000)
parser.add_argument("--warm_up", type=int, default=10000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--update_target", type=int, default=1000)

args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_agent(args, device)
