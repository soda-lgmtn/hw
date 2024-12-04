import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import math
from enum import Enum
import random
import csv
from datetime import datetime

class DriveMode(Enum):
    NORMAL = 1
    AVOID_OBSTACLE = 2
    EMERGENCY_STOP = 3
    RANDOM_EXPLORE = 4
    WANDERING = 5

class PerformanceMetrics:
    def __init__(self):
        self.total_distance = 0.0          # Total distance traveled
        self.collision_count = 0           # Number of collisions
        self.start_time = time.time()      # Start time
        self.last_position = None          # Last position
        self.stuck_count = 0               # Number of times stuck
        self.stuck_threshold = 0.1         # Threshold for determining stuck
        self.stuck_time_threshold = 2.0    # Threshold for determining stuck
        self.last_movement_time = time.time()
        self.emergency_stops = 0           # Number of emergency stops
        self.smooth_steering_score = 0     # Steering smoothness score
        self.last_steering = 0             # Last steering angle
        self.total_steering_changes = 0    # Number of steering changes
        self.metrics_history = []  # Used to store historical data
        
    def update_metrics(self, car_state, current_mode, steering_angle):
        current_position = car_state['position']
        current_time = time.time()

        # Update total distance
        if self.last_position is not None:
            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(current_position, self.last_position))
            )
            self.total_distance += distance

            # Check if stuck
            if distance < self.stuck_threshold:
                if current_time - self.last_movement_time > self.stuck_time_threshold:
                    self.stuck_count += 1
            else:
                self.last_movement_time = current_time

        self.last_position = current_position

        # Update steering smoothness
        steering_change = abs(steering_angle - self.last_steering)
        self.total_steering_changes += steering_change
        self.smooth_steering_score = 1.0 / (1.0 + self.total_steering_changes)
        self.last_steering = steering_angle

        # Update mode-related metrics
        if current_mode == DriveMode.EMERGENCY_STOP:
            self.emergency_stops += 1

    def check_collision(self, contact_points):
        """Check for collisions"""
        if len(contact_points) > 0:
            self.collision_count += 1
            return True
        return False

    def get_performance_report(self):
        """Generate performance report"""
        runtime = time.time() - self.start_time
        return {
            "Total Distance (m)": f"{self.total_distance:.2f} meters",
            "Average Speed (m/s)": f"{self.total_distance/runtime:.2f} meters/second",
            "Collision Count": self.collision_count,
            "Stuck Count": self.stuck_count,
            "Emergency Stops": self.emergency_stops,
            "Steering Smoothness": f"{self.smooth_steering_score:.2f}",
            "Runtime (s)": f"{runtime:.2f} seconds"
        }

    def save_to_csv(self):
        """Save performance metrics to a CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_metrics_{timestamp}.csv"
        
        # Get the final performance report
        final_metrics = self.get_performance_report()
        
        # Prepare CSV data
        fieldnames = [
            "Timestamp",
            "Total Distance (m)",
            "Average Speed (m/s)",
            "Collision Count",
            "Stuck Count",
            "Emergency Stops",
            "Steering Smoothness",
            "Runtime (s)"
        ]
        
        # Process data, remove units and other strings
        csv_data = {
            "Timestamp": timestamp,
            "Total Distance (m)": float(final_metrics["Total Distance (m)"].split()[0]),
            "Average Speed (m/s)": float(final_metrics["Average Speed (m/s)"].split()[0]),
            "Collision Count": final_metrics["Collision Count"],
            "Stuck Count": final_metrics["Stuck Count"],
            "Emergency Stops": final_metrics["Emergency Stops"],
            "Steering Smoothness": float(final_metrics["Steering Smoothness"]),
            "Runtime (s)": float(final_metrics["Runtime (s)"].split()[0])
        }
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(csv_data)
            print(f"\nPerformance metrics saved to: {filename}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

class AutonomousCar:
    def __init__(self):
        # Connect to the physics engine
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1./240.)
        
        # Set up the environment
        self.setup_environment()
        
        random_yaw = random.uniform(0, 2 * math.pi)
        random_orientation = p.getQuaternionFromEuler([0, 0, random_yaw])
        
        self.car = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2], random_orientation)
        
        # Print joint information
        num_joints = p.getNumJoints(self.car)
        print(f"Number of joints: {num_joints}")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.car, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
        
        # Set up camera parameters
        self.setup_camera()
        
        # Set up lidar parameters
        self.setup_lidar()
        
        # Motion control parameters
        self.setup_motion_parameters()
        
        # Obstacle avoidance parameters
        self.setup_avoidance_parameters()
        
        # PID controller parameters
        self.setup_pid_controller()
        
        # Random motion parameters
        self.setup_random_motion_parameters()
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
    def setup_motion_parameters(self):
        """Set motion control parameters"""
        self.max_speed = 100.0
        self.min_speed = 80.0
        self.max_steering = 0.5
        self.current_speed = 0.0
        self.current_steering = 0.0
        
        # According to the actual joint configuration of racecar.urdf
        self.wheel_indices = [2, 3, 5, 7]  # Rear wheels and front wheels
        self.steering_indices = [4, 6]      # Steering joints
        
    def setup_avoidance_parameters(self):
        """Set obstacle avoidance parameters"""
        self.safe_distance = 2.5  # Safe distance
        self.emergency_distance = 1.5  # Emergency braking distance
        self.current_mode = DriveMode.NORMAL
        self.avoid_time = 0
        self.last_mode_change = time.time()
        
    def setup_random_motion_parameters(self):

        self.random_direction_interval = 3.0 
        self.last_direction_change = time.time()
        self.current_random_target = 0.0
        self.wandering_amplitude = 0.5
        self.wandering_frequency = 0.2
        self.wander_time = 0
        
    def setup_environment(self):
        """Set up the environment"""
        # Set gravity and ground
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Create boundaries
        self.create_boundary()
        # Add random obstacles
        self.create_obstacles()
        
    def create_boundary(self):
        """Create environment boundaries"""
        wall_height = 1.0
        wall_thickness = 0.2
        arena_size = 20.0
        
        # Create four walls
        walls = [
            # North wall
            ([0, arena_size/2, wall_height/2], [arena_size, wall_thickness, wall_height]),
            # South wall
            ([0, -arena_size/2, wall_height/2], [arena_size, wall_thickness, wall_height]),
            # East wall
            ([arena_size/2, 0, wall_height/2], [wall_thickness, arena_size, wall_height]),
            # West wall
            ([-arena_size/2, 0, wall_height/2], [wall_thickness, arena_size, wall_height])
        ]
        
        for position, size in walls:
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                rgbaColor=[0.7, 0.7, 0.7, 1]
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
            
    def create_obstacles(self):
        """Create random obstacles"""
        num_obstacles = 15  
        arena_size = 18.0  
        min_distance = 3.0  
        
        obstacle_positions = []
        
        for _ in range(num_obstacles):
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 50:
                # Generate a random position
                x = random.uniform(-arena_size/2, arena_size/2)
                y = random.uniform(-arena_size/2, arena_size/2)
                
                # Check the distance from other obstacles
                valid_position = True
                for pos in obstacle_positions:
                    if math.sqrt((x - pos[0])**2 + (y - pos[1])**2) < min_distance:
                        valid_position = False
                        break
                
                if math.sqrt(x**2 + y**2) < 5.0:  # Ensure not too close to the starting point
                    valid_position = False
                
                attempts += 1
            
            if valid_position:
                obstacle_positions.append((x, y))
                
                size = random.uniform(0.8, 2.0)
                color = [random.random(), random.random(), random.random(), 1]
                
                # Create a random shape obstacle
                shape_type = random.choice([p.GEOM_BOX, p.GEOM_CYLINDER])
                if shape_type == p.GEOM_BOX:
                    visual_shape = p.createVisualShape(
                        shapeType=shape_type,
                        halfExtents=[size/2, size/2, size],
                        rgbaColor=color
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=shape_type,
                        halfExtents=[size/2, size/2, size]
                    )
                else:
                    visual_shape = p.createVisualShape(
                        shapeType=shape_type,
                        radius=size/2,
                        length=size*2,
                        rgbaColor=color
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=shape_type,
                        radius=size/2,
                        height=size*2
                    )
                
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, size],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, random.uniform(0, 2*math.pi)])
                )

    def setup_camera(self):
        """Set camera parameters"""
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90
        self.camera_aspect = float(self.camera_width) / float(self.camera_height)
        self.camera_near = 0.1
        self.camera_far = 20
        # Create an OpenCV window
        cv2.namedWindow('Car Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Car Camera View', 640, 480)

    def get_camera_image(self):
        """Get the vehicle's camera image"""
        # Get the vehicle's position and orientation
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car)
        
        camera_height = 0.3
        camera_forward = 0.3
        rot_matrix = p.getMatrixFromQuaternion(car_orn)
        camera_pos = [
            car_pos[0] + rot_matrix[0] * camera_forward,
            car_pos[1] + rot_matrix[3] * camera_forward,
            car_pos[2] + camera_height
        ]
        
        # Calculate the target point (in front of the car)
        target_pos = [
            camera_pos[0] + rot_matrix[0],
            camera_pos[1] + rot_matrix[3],
            camera_pos[2] - 0.1  # Slightly downward
        ]
        
        # Calculate the view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        # Calculate the projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        
        # Get the image
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to OpenCV format
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]  # Remove the alpha channel
        
        # Display the image
        cv2.imshow('Car Camera View', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        return rgb_array, depth_img

    def setup_lidar(self):
        """Set lidar parameters"""
        self.lidar_range = 10.0  # Increase the detection range
        self.lidar_resolution = 360
        self.sector_size = 30  # Sector size (degrees)
        self.num_sectors = 360 // self.sector_size
        
    def setup_pid_controller(self):
        """Set PID controller parameters"""
        self.steering_pid = {
            'P': 1.0,    # Increase P value
            'I': 0.01,   # Decrease I value
            'D': 0.1,    # Decrease D value
            'prev_error': 0,
            'integral': 0
        }
        self.speed_pid = {
            'P': 1.0,    # Increase P value
            'I': 0.01,   # Decrease I value
            'D': 0.1,    # Decrease D value
            'prev_error': 0,
            'integral': 0
        }
        
    def get_car_state(self):
        """Get the current state of the car"""
        position, orientation = p.getBasePositionAndOrientation(self.car)
        linear_vel, angular_vel = p.getBaseVelocity(self.car)
        return {
            'position': position,
            'orientation': orientation,
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel
        }
        
    def get_lidar_data(self):
        """Get lidar data and perform sector analysis"""
        car_state = self.get_car_state()
        car_pos = car_state['position']
        car_orn = car_state['orientation']
        
        # Get the vehicle's heading
        _, _, yaw = p.getEulerFromQuaternion(car_orn)
        
        lidar_data = []
        sector_data = [float('inf')] * self.num_sectors
        
        for angle in range(self.lidar_resolution):
            # Calculate the ray direction in the global coordinate system
            global_angle = angle * (2 * np.pi / self.lidar_resolution) + yaw
            direction = [np.cos(global_angle), np.sin(global_angle), 0]
            
            # Shoot a ray
            ray_to = [
                car_pos[0] + direction[0] * self.lidar_range,
                car_pos[1] + direction[1] * self.lidar_range,
                car_pos[2]
            ]
            
            result = p.rayTest(car_pos, ray_to)[0]
            distance = result[2] * self.lidar_range
            
            lidar_data.append(distance)
            
            # Update sector data
            sector_index = angle // self.sector_size
            sector_data[sector_index] = min(sector_data[sector_index], distance)
            
        return lidar_data, sector_data
        
    def analyze_environment(self):
        """Analyze the environment and decide on an action strategy"""
        lidar_data, sector_data = self.get_lidar_data()
        
        # 扩大检测范围，提前发现障碍物
        front_sector = 0  # 正前方扇区索引
        left_sectors = list(range(1, 6))  # 左侧扇区索引
        right_sectors = list(range(6, 11))  # 右侧扇区索引
        
        # 获取各个方向的最小距离
        front_distance = sector_data[front_sector]
        left_distances = [sector_data[i] for i in left_sectors]
        right_distances = [sector_data[i] for i in right_sectors]
        
        # 计算左右两侧的平均距离，用于决定转向方向
        left_avg_distance = sum(left_distances) / len(left_distances)
        right_avg_distance = sum(right_distances) / len(right_distances)
        
        # 提高安全距离和紧急距离的阈值
        self.safe_distance = 4.0  # 增加安全距离
        self.emergency_distance = 2.0  # 增加紧急制动距离
        
        # 根据障碍物距离决定行为模式
        if front_distance < self.emergency_distance:
            # 紧急情况：立即停止并快速转向空旷方向
            if left_avg_distance > right_avg_distance:
                return DriveMode.EMERGENCY_STOP, -self.max_steering
            else:
                return DriveMode.EMERGENCY_STOP, self.max_steering
            
        elif front_distance < self.safe_distance:
            if left_avg_distance > right_avg_distance:
                # 向左转，转向角度与障碍物距离成反比
                steering = -self.max_steering * (1.5 - front_distance/self.safe_distance)
                return DriveMode.AVOID_OBSTACLE, steering
            else:
                # 向右转，转向角度与障碍物距离成反比
                steering = self.max_steering * (1.5 - front_distance/self.safe_distance)
                return DriveMode.AVOID_OBSTACLE, steering
        else:
            # 安全距离内无障碍物，进行随机探索
            # 减小随机转向的频率和幅度，使运动更平滑
            if time.time() - self.last_direction_change > self.random_direction_interval:
                # 根据左右空间的差异选择随机转向的范围
                if abs(left_avg_distance - right_avg_distance) > 2.0:
                    # 倾向于向空旷方向转向
                    if left_avg_distance > right_avg_distance:
                        self.current_random_target = random.uniform(-self.max_steering/2, 0)
                    else:
                        self.current_random_target = random.uniform(0, self.max_steering/2)
                else:
                    # 空间相近时小角度随机转向
                    self.current_random_target = random.uniform(-self.max_steering/4, self.max_steering/4)
                self.last_direction_change = time.time()
            
            return DriveMode.RANDOM_EXPLORE, self.current_random_target
            
    def move_car(self, speed, steering_angle):
        """Control the movement of the car"""
        # Debug output
        print(f"Applying speed: {speed}, steering: {steering_angle}")
        print("Wheel indices:", self.wheel_indices)
        print("Steering indices:", self.steering_indices)
        
        for wheel_idx in self.wheel_indices:
            try:
                p.setJointMotorControl2(
                    bodyUniqueId=self.car,
                    jointIndex=wheel_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=speed,
                    force=400.0
                )
            except Exception as e:
                print(f"Error setting wheel {wheel_idx}: {e}")
        
        # Apply steering
        for steering_idx in self.steering_indices:
            try:
                p.setJointMotorControl2(
                    bodyUniqueId=self.car,
                    jointIndex=steering_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=steering_angle,
                    force=100.0
                )
            except Exception as e:
                print(f"Error setting steering {steering_idx}: {e}")

    def update_motion(self):
        """Update the motion state"""
        # Get the current state
        car_state = self.get_car_state()
        
        # Check for collisions
        contact_points = p.getContactPoints(self.car)
        if self.metrics.check_collision(contact_points):
            print("Collision detected!")
        
        # Analyze the environment and get the action mode
        mode, turn_direction = self.analyze_environment()
        
        # Set the target speed and steering based on the mode
        if mode == DriveMode.EMERGENCY_STOP:
            target_speed = 0
            target_steering = 0
        elif mode == DriveMode.AVOID_OBSTACLE:
            target_speed = self.min_speed
            target_steering = turn_direction
        elif mode == DriveMode.RANDOM_EXPLORE:
            target_speed = self.max_speed * 0.5
            target_steering = turn_direction
        else:
            target_speed = self.max_speed
            target_steering = 0
        
        # Ensure there is always a base speed
        target_speed = max(self.min_speed, target_speed)
        
        # Directly apply speed and steering
        self.current_speed = target_speed
        self.current_steering = target_steering
        
        # Control the car
        self.move_car(self.current_speed, self.current_steering)
        
        # Update performance metrics
        self.metrics.update_metrics(car_state, mode, self.current_steering)

def main():
    car = AutonomousCar()
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    try:
        while True:
            car.update_motion()
            p.stepSimulation()
            car.get_camera_image()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nPerformance Evaluation Report:")
        for metric, value in car.metrics.get_performance_report().items():
            print(f"{metric}: {value}")
            
        car.metrics.save_to_csv()
    
        p.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()