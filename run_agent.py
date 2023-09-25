import numpy as np

from env import Environment
from agent_brain import QLearningTable

def calculate_metrics(steps, costs, total_episodes, exploration_count):
    # Calculate average episode length
    avg_episode_length = np.mean(steps)

    # Calculate average episode cost
    avg_episode_cost = np.mean(costs)

    # Calculate success rate (percentage of episodes that reached the goal)
    success_rate = (steps.count(10) / len(steps)) * 100

    # Calculate exploration rate (percentage of exploration actions)
    exploration_rate = (exploration_count / total_episodes) * 100

    # Print the calculated metrics
    print("Average Episode Length:", avg_episode_length)
    print("Average Episode Cost:", avg_episode_cost)
    print("Success Rate (%):", success_rate)
    print("Exploration Rate (%):", exploration_rate)
    print("Total Episodes:", total_episodes)

def run_episodes():
    episode_steps = []  # List to store the number of steps for each episode
    episode_costs = []  # List to store the cost for each episode

    for episode in range(1000):
        agent.update_episode_count()  # Increment episode count at the beginning of each episode
        # Reset the environment to the initial observation
        observation = env.reset()

        # Initialize episode-specific variables
        num_steps = 0
        episode_cost = 0

        while True:
            # Render the environment
            env.render()

            # Agent chooses an action based on the current observation
            action = agent.choose_action(str(observation))

            # Agent takes an action and receives the next observation and reward
            next_observation, reward, done = env.step(action)

            # Agent learns from this transition and calculates the cost
            episode_cost += agent.learn(str(observation), action, reward, str(next_observation))

            # Update the current observation
            observation = next_observation

            # Increment the step count
            num_steps += 1

            # Break the loop when the episode ends (agent reaches the goal or an obstacle)
            if done:
                episode_steps.append(num_steps)
                episode_costs.append(episode_cost)
                break
    # Calculate metrics
    calculate_metrics(episode_steps, episode_costs, agent.total_episodes, agent.exploration_count)

    # Show the final route
    env.final()

    # Show the Q-table with values for each action
    agent.print_q_table()

    # Plot the results
    agent.plot_results(episode_steps, episode_costs)

# Entry point
if __name__ == "__main__":
    # Create the environment
    env = Environment()

    # Create the Q-learning agent
    agent = QLearningTable(actions=list(range(env.n_actions)))

    # Start running episodes
    env.after(100, run_episodes)
    env.mainloop()
