# ==============================================================================
# ==============================================================================

# Hey, I modified Copyright 2025 DeepMind Technologies Limited
# for an upcoming project regarding navigation policies for Go2 Unitree's quadrupedal robot.

# ==============================================================================
# ==============================================================================
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import time
from mujoco_playground.experimental.sim2sim.Dimitris.LocomotionPolicy import LocomotionPolicy
from mujoco_playground.experimental.sim2sim.Dimitris.TrajectoryGenerator import TrajectoryGenerator

class TrajectoryPolicy:
    def __init__(
        self,
        n_substeps: int = 5,
        locomotion_policy: LocomotionPolicy = None,
        trajectory_generator: TrajectoryGenerator = None,
    ):
        self._counter = 0
        self._n_substeps = n_substeps
        self.locomotion_policy = locomotion_policy
        self.trajectory_generator = trajectory_generator
        self.target = None
        self.init_pos = None
                
        self.current_pos = np.zeros(2)
        self.current_vel = np.zeros(2)

        self.last_update_time = time.time()  # initialize properly
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0

    def update_control(self):
        # Generate random velocity
        self.generate_velocity() #sets a random veliocity --> change with a new trajectory
        
        self.locomotion_policy.set_cmd_vel(self.cmd_vel_x, self.cmd_vel_y) #updates the random velocity --> change with the required velocities needed to trajectory

    def update_trajectory_control(self):
        
        self.locomotion_policy.set_cmd_vel(self.cmd_vel_x, self.cmd_vel_y)
        
    def random_controller(self, model, data):
        """Set random velocity commands"""
        current_time = time.time()


        if (current_time - self.last_update_time) >= 2.0: # 2 second has passed
            self.update_control()
            self.last_update_time = current_time
        # Always apply control
        if self.locomotion_policy:
            self.locomotion_policy.get_control(model, data)


    def trajectory_controller(self, model, data):
        '''guides the robot to follow a trajectory'''
        #set the control loop
        self._update_current_state(data)

        # Check if a target exists, if not, create one
        if self.target is None:
            print("Setting initial target.")
            self.set_target()

        # Check if the robot has reached the target
        error_pos_norm = np.linalg.norm(self.current_pos - self.target)
        if error_pos_norm < 0.2:  # Increased threshold for smoother transitions
            print(f"Target reached. Current position: {self.current_pos}")
            self.set_target()
        
        # Compute the desired velocity command using the PD controller
        self.cmd_vel_x, self.cmd_vel_y = self.compute_pd_velocity()
        
        # Send the computed velocity to the low-level locomotion policy
        self.locomotion_policy.set_cmd_vel(self.cmd_vel_x, self.cmd_vel_y)
        
        # Finally, execute the low-level control step
        if self.locomotion_policy:
            self.locomotion_policy.get_control(model, data)
            
    def compute_pd_velocity(self):
        """PD controller to compute velocity command towards the target."""
        
        # Controller gains - you can tune these
        Kp = 2.5  
        Kd = 0.3  

       
        pos_error = self.target - self.current_pos
        p_term = Kp * pos_error

       
        d_term = Kd * self.current_vel
      
        vel_cmd = p_term - d_term

        # Clip the final command to be within a reasonable range (e.g., [-1, 1])
        vel_cmd = np.clip(vel_cmd, -1.0, 1.0)
        
        return vel_cmd[0], vel_cmd[1]
                
            
    def compute_velocity(self,data):
        """Simple PD controller to compute velocity command towards the target."""

        # Get current position and velocity from locomotion policy
        
        Kp = 10.0
        Kd = 0.2

        # Position error
        pos_error = np.array(self.target) - self.current_pos

        # Velocity command using PD
        vel_cmd = Kp * pos_error

        # Optional: Clip velocities to [-1, 1]
        vel_cmd = np.clip(vel_cmd, -1.0, 1.0).flatten()
        
        #set the cmd_vel
        self.cmd_vel_x = vel_cmd[0]
        self.cmd_vel_y = vel_cmd[1]

        return vel_cmd[0], vel_cmd[1]
    
    
    def generate_velocity(self):
        
        self.cmd_vel_x = np.around(np.random.uniform(-1, 1), 3)
        self.cmd_vel_y = np.around(np.random.uniform(-1, 1), 3)
        
    def set_target(self):
        # Set the target for the locomotion policy
        # In the set_target method...
        self.target = np.array(self.trajectory_generator._generate_simple_target()).flatten()
        
    def _update_current_state(self, data):
        """Fetches the robot's current position and velocity."""
        
        # Get position (assuming the method returns [x, y, z] or similar)
        full_pos = self.locomotion_policy.current_pos(data)
        self.current_pos = np.array(full_pos[:2])

       
        full_vel = self.locomotion_policy.current_vel(data)
        self.current_vel = np.array(full_vel[:2])

    def get_velocity(self):
        return self.cmd_vel_x, self.cmd_vel_y