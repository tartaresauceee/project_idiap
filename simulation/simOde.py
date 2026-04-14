import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


class ImpedanceSimulator:
    """Simulate 2D impedance-controlled motion with wall contact."""
    
    def __init__(self, duration=2.0, dt=1e-3):
        """Initialize simulator with parameters."""
        # Time parameters
        self.duration = duration
        self.dt = dt
        self.t = np.arange(0, duration, dt)
        
        # Impedance parameters
        self.M = np.eye(2) * 0.1
        self.K = np.array([[150, 0], [0, 150]])
        self.B = 2 * np.sqrt(np.multiply(self.M, self.K))
        
        # Wall parameters
        self.K_wall = 1000
        self.B_wall = 0.1 * self.K_wall
        self.wall_y = 0.0
        
        # Goal trajectory
        self.x0 = np.array([0.0, 0.02])
        self.x_goal = np.array([0.0, -0.005])
        
        # Initial conditions
        self.initial_state = np.concatenate([self.x0, np.zeros(2)])
        
        # Results
        self.solution = None
        self.wall_force_hist = []
    
    def min_jerk(self, t):
        """Compute 2D minimum jerk trajectory at time t."""
        if t > self.duration:
            return self.x_goal
        
        if t < 0:
            return self.x0

        # Motion towards surface
        if t <= self.duration/2:
            A = self.x_goal - self.x0
            term1 = 10 / (self.duration/2)**3 * t**3
            term2 = -15 / (self.duration/2)**4 * t**4
            term3 = 6 / (self.duration/2)**5 * t**5

            return self.x0 + A * (term1 + term2 + term3)

        # Motion away from surface
        else:
            A = self.x0 - self.x_goal
            term1 = 10 / (self.duration/2)**3 * (t - self.duration/2)**3
            term2 = -15 / (self.duration/2)**4 * (t - self.duration/2)**4
            term3 = 6 / (self.duration/2)**5 * (t - self.duration/2)**5

            return self.x_goal + A * (term1 + term2 + term3)
    
    def wall_force(self, x, x_dot):
        """Compute wall contact force."""
        if self.wall_y - x[1] <= 0:
            return np.array([0.0, 0.0])
        
        f_y = self.K_wall * (self.wall_y - x[1]) #+ self.B_wall * (-x_dot[1])
        return np.array([0.0, f_y])
    
    def compute_wall_forces(self):
        self.wall_force_hist = []
        for i in range(len(self.solution)):
            x = self.solution[i, :2]
            x_dot = self.solution[i, 2:]
            f_wall = self.wall_force(x, x_dot)
            self.wall_force_hist.append(f_wall)
    
        self.wall_force_hist = np.array(self.wall_force_hist)  # shape (len(t), 2)
    
    def acceleration(self, x, x_dot, t, f_external):
        """Compute acceleration from impedance equation."""
        x_zft = self.min_jerk(t)
        accel = np.linalg.inv(self.M) @ (
            self.B @ (-x_dot) + self.K @ (x_zft - x) + f_external
        )
        return accel
    
    def compute_zft_hat(self):
        force = - self.wall_force_hist[:,1]
        K = self.K[1, 1]
        x2 = self.solution[:,1]

        self.zft_hat = 1./K * force + x2
    
    def system(self, t, state):
        """ODE system: convert 2nd order to 1st order ODEs."""
        x = state[:2]
        x_dot = state[2:]
        
        f_external = self.wall_force(x, x_dot)
        x_dotdot = self.acceleration(x, x_dot, t, f_external)
        
        return np.concatenate([x_dot, x_dotdot])
    
    def simulate(self, method):
        """Run the simulation."""

        if method == "odeint":
            # odeint needs f(y, t) — flip arguments
            fun_flipped = lambda y, t: self.system(t, y)
            
            self.solution = odeint(fun_flipped, self.initial_state, self.t)
            
            return
        
        else:
            # solve_ivp methods: RK45, RK23, DOP853, Radau, BDF, LSODA
            t_span = [0, self.duration]
            sol = solve_ivp(self.system, t_span, self.initial_state, method=method,
                            t_eval=self.t)
            
            self.solution = sol.y.T
        
        return
    
    def plot_results(self):
        """Plot simulation results."""
        if self.solution is None:
            print("Run simulate() first")
            return
        
        x2 = self.solution[:, 1]
        self.zft = np.array([self.min_jerk(ti) for ti in self.t])

        rmse = np.sqrt(1/len(self.zft[:,1]) * np.sum((self.zft[:,1] - self.zft_hat)**2))
        
        fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        
        # Left y-axis: Position
        ax[0].plot(self.t, x2*1e3, color="blue", label='Y (actual)')
        ax[0].plot(self.t, self.zft[:, 1]*1e3, color='orange', label='Y (ZFT)')
        ax[0].plot(self.t, self.zft_hat*1e3, color='cyan', label=f'Reconstructed ZFT') #  (RMSE: {rmse:.4f}m)
        ax[0].hlines(self.wall_y, 0, self.t[-1], colors='black', label='Wall', linestyles='--')
        ax[0].set_ylabel('Position (mm)', color='blue', size='large')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc='lower right')
        
        # Right y-axis: Force

        ax[1].plot(self.t, self.wall_force_hist[:,1], color="red", label='Vertical wall force')
        ax[1].set_xlabel('Time (s)', size='large')
        ax[1].set_ylabel('Force (N)', color='red', size='large')
        ax[1].legend(loc='upper right')

        fig.suptitle("Substractive Impedance", size='x-large')
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Create simulator with custom parameters
    sim = ImpedanceSimulator(duration=5.0, dt=1e-3)
    
    # Run simulation
    sim.simulate("RK45")

    # Compute wall forces
    sim.compute_wall_forces()

    # Compute ZFT estimate
    sim.compute_zft_hat()
    
    # Plot results
    sim.plot_results()

