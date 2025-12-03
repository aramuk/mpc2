import numpy as np
from scipy.optimize import minimize

class MPCGridNavigator:
    def __init__(self, grid_size=10, horizon=5, dt=1.0):
        self.grid_size = grid_size
        self.horizon = horizon  # prediction horizon
        self.dt = dt
        
        # Cost weights
        self.Q = np.diag([10.0, 10.0])  # state cost (position)
        self.R = np.diag([1.0, 1.0])    # control cost (velocity)
        
    def dynamics(self, state, control):
        """Simple discrete-time dynamics: position += velocity * dt"""
        x, y = state
        vx, vy = control
        return np.array([x + vx * self.dt, y + vy * self.dt])
    
    def cost_function(self, u_flat, x0, goal, obstacles):
        """Total cost over horizon"""
        u = u_flat.reshape((self.horizon, 2))
        cost = 0.0
        x = x0.copy()
        
        for t in range(self.horizon):
            # Predict next state
            x = self.dynamics(x, u[t])
            
            # Obstacle cost (penalize collisions)
            if obstacles is not None:
                for obstacle in obstacles:
                    if np.linalg.norm(x - obstacle) < 1.0:
                        cost += 1e10
                        return cost

            # State cost (distance to goal)
            state_error = x - goal
            print(state_error.T.shape, self.Q.shape, state_error.shape)
            cost += state_error.T @ self.Q @ state_error
            
            # Control cost (penalize large velocities) 
            cost += u[t].T @ self.R @ u[t]
            
        return cost
    
    def solve(self, current_pos, goal_pos, obstacles):
        """Solve MPC optimization problem"""
        # Initial guess (zero controls)
        u0 = np.zeros(self.horizon * 2)
        
        # Bounds on control (velocity limits)
        bounds = [(-2.0, 2.0)] * (self.horizon * 2)
        
        # Optimize
        result = minimize(
            self.cost_function,
            u0,
            args=(current_pos, goal_pos, obstacles),
            method='SLSQP',
            bounds=bounds
        )
        
        # Return first control action
        u_optimal = result.x.reshape((self.horizon, 2))
        return u_optimal[0]

# Example usage
def main():
    mpc = MPCGridNavigator(grid_size=10, horizon=5)
    
    # Initial position and goal
    pos = np.array([1.0, 1.0])
    goal = np.array([-20.0, 35.0])
    # Diagonal obstacles
    obstacles = np.ones((1, 2)) * np.linspace(0, 50, 50).reshape(-1, 1) + 2.0

    # Get stuck because box around initial position
    # obstacles = np.array([[0.,0],[0,1.],[0, 2.],[1, 0.],[1, 2.],[2, 0.],[2, 1.],[2, 2.]])

    trajectory = [pos.copy()]
    
    # Simulate navigation
    for step in range(20):
        # Compute optimal control
        control = mpc.solve(pos, goal, obstacles)
        
        # Apply control and update position
        pos = mpc.dynamics(pos, control)
        trajectory.append(pos.copy())
        
        print(f"Step {step}: pos=({pos[0]:.2f}, {pos[1]:.2f}), control=({control[0]:.2f}, {control[1]:.2f})")
        
        # Check if reached goal
        if np.linalg.norm(pos - goal) < 0.5:
            print("Goal reached!")
            break
    
    # Print trajectory
    print("\nTrajectory:")
    for i, p in enumerate(trajectory):
        print(f"  {i}: ({p[0]:.2f}, {p[1]:.2f})")

if __name__ == "__main__":
    main()