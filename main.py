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
        
        # Generate a random direction
        random_yaw = random.uniform(0, 2 * math.pi)
        random_orientation = p.getQuaternionFromEuler([0, 0, random_yaw])
        
        # Load the car and set a random direction
        self.car = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2], random_orientation)
        
        # Setup the environment
        self.setup_environment()
        
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
        """Set random motion parameters"""
        self.random_direction_interval = 3.0  # Time interval to change direction
        self.last_direction_change = time.time()
        self.current_random_target = 0.0
        self.wandering_amplitude = 0.5
        self.wandering_frequency = 0.2
        self.wander_time = 0
        
    def setup_environment(self):
        """设置环境"""
        # 设置重力和地面
        p.setGravity(0, 0, -9.81)
        
        # 加载新的地面模型（使用更复杂的地形）
        ground_plane = p.loadURDF("plane.urdf")
        
        # 创建一个封闭的竞技场
        self.create_arena()
        # 添加随机障碍物和地形特征
        self.create_terrain_features()
        
    def create_arena(self):
        """创建竞技场边界和基本结构"""
        arena_size = 30.0  # 更大的竞技场
        wall_height = 2.0
        wall_thickness = 0.2
        
        # 创建围墙
        walls = [
            # 北墙
            ([0, arena_size/2, wall_height/2], [arena_size, wall_thickness, wall_height]),
            # 南墙
            ([0, -arena_size/2, wall_height/2], [arena_size, wall_thickness, wall_height]),
            # 东墙
            ([arena_size/2, 0, wall_height/2], [wall_thickness, arena_size, wall_height]),
            # 西墙
            ([-arena_size/2, 0, wall_height/2], [wall_thickness, arena_size, wall_height])
        ]
        
        for position, size in walls:
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                rgbaColor=[0.5, 0.5, 0.5, 1]
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

    def create_terrain_features(self):
        """创建地形特征和障碍物"""
        arena_size = 28.0  # 略小于竞技场大小
        
        # 创建随机分布的障碍物组
        self.create_obstacle_clusters()
        
        # 创建蛇形通道
        self.create_snake_path()
        
        # 创建坡道和台阶
        self.create_ramps_and_steps()

    def create_obstacle_clusters(self):
        num_clusters = 8
        obstacles_per_cluster = 6
        arena_size = 28.0
        safe_radius = 3.0  # 小车周围的安全半径
        
        car_pos, _ = p.getBasePositionAndOrientation(self.car)
        
        for _ in range(num_clusters):
            cluster_x = random.uniform(-arena_size/2 + 5, arena_size/2 - 5)
            cluster_y = random.uniform(-arena_size/2 + 5, arena_size/2 - 5)
            
            # 检查与小车的距离
            distance_to_car = math.sqrt(
                (cluster_x - car_pos[0])**2 + 
                (cluster_y - car_pos[1])**2
            )
            
            # 如果太靠近小车，跳过这个簇
            if distance_to_car < safe_radius:
                continue
            
            # 在簇周围创建障碍物
            for _ in range(obstacles_per_cluster):
                offset_x = random.uniform(-2, 2)
                offset_y = random.uniform(-2, 2)
                x = cluster_x + offset_x
                y = cluster_y + offset_y
                
                # 再次检查具体障碍物位置与小车的距离
                distance_to_car = math.sqrt(
                    (x - car_pos[0])**2 + 
                    (y - car_pos[1])**2
                )
                
                if distance_to_car < safe_radius:
                    continue
                
                size = random.uniform(0.5, 1.5)
                height = random.uniform(0.5, 2.0)
                color = [random.random(), random.random(), random.random(), 1]
                
                # 随机选择障碍物形状
                shape_type = random.choice([p.GEOM_CYLINDER, p.GEOM_BOX])
                if shape_type == p.GEOM_CYLINDER:
                    visual_shape = p.createVisualShape(
                        shapeType=shape_type,
                        radius=size/2,
                        length=height,
                        rgbaColor=color
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=shape_type,
                        radius=size/2,
                        height=height
                    )
                else:
                    visual_shape = p.createVisualShape(
                        shapeType=shape_type,
                        halfExtents=[size/2, size/2, height/2],
                        rgbaColor=color
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=shape_type,
                        halfExtents=[size/2, size/2, height/2]
                    )
                
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, height/2]
                )

    def create_snake_path(self):
        """创建蛇形通道"""
        path_length = 20
        path_width = 3
        wall_height = 1.5
        safe_radius = 3.0  # 小车周围的安全半径
        
        # 获取小车的初始位置
        car_pos, _ = p.getBasePositionAndOrientation(self.car)
        
        # 创建蛇形路径的墙壁
        for i in range(4):
            x = -10 + i * 5
            y = 5 * math.sin(i * math.pi / 2)
            
            # 检查与小车的距离
            distance_to_car = math.sqrt(
                (x - car_pos[0])**2 + 
                (y - car_pos[1])**2
            )
            
            # 如果太靠近小车，跳过这个墙段
            if distance_to_car < safe_radius:
                continue
            
            # 创建通道墙
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.2, path_width, wall_height],
                rgbaColor=[0.7, 0.7, 0.2, 1]
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.2, path_width, wall_height]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, wall_height/2]
            )

    def create_ramps_and_steps(self):

        ramp_length = 4
        ramp_width = 2
        ramp_height = 1
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[ramp_length/2, ramp_width/2, ramp_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[ramp_length/2, ramp_width/2, ramp_height/2]
        )
        
        # 在不同位置放置斜坡
        ramp_positions = [
            ([8, 8, ramp_height/2], [0, 0, math.pi/6]),  # 30度斜坡
            ([-8, -8, ramp_height/2], [0, 0, -math.pi/6])  # -30度斜坡
        ]
        
        for pos, orn in ramp_positions:
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos,
                baseOrientation=p.getQuaternionFromEuler(orn)
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
        
        # Analyze the obstacle situation in each direction
        front_distance = min(sector_data[0], sector_data[-1])  # Front sector
        left_distances = sector_data[2:4]  # Left sector
        right_distances = sector_data[8:10]  # Right sector
        
        # Determine the current mode
        if front_distance < self.emergency_distance:
            return DriveMode.EMERGENCY_STOP, 0
        elif front_distance < self.safe_distance:
            # Choose the safer steering direction
            left_space = min(left_distances)
            right_space = min(right_distances)
            if left_space > right_space:
                return DriveMode.AVOID_OBSTACLE, -self.max_steering  # Turn left
            else:
                return DriveMode.AVOID_OBSTACLE, self.max_steering   # Turn right
        else:
            # Random exploration mode
            if time.time() - self.last_direction_change > self.random_direction_interval:
                self.current_random_target = random.uniform(-self.max_steering, self.max_steering)
                self.last_direction_change = time.time()
            return DriveMode.RANDOM_EXPLORE, self.current_random_target
            
    def move_car(self, speed, steering_angle):
        """Control the movement of the car"""
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
    # Create an instance of the autonomous car
    car = AutonomousCar()
    
    # Give some time for the physics engine to stabilize
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Run the simulation
    try:
        while True:
            car.update_motion()
            p.stepSimulation()
            car.get_camera_image()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        # Print performance report
        print("\nPerformance Evaluation Report:")
        for metric, value in car.metrics.get_performance_report().items():
            print(f"{metric}: {value}")
            
        # Save performance metrics to CSV file
        car.metrics.save_to_csv()
        
        # Clean up resources
        p.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()