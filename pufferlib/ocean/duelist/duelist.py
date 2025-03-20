import pettingzoo
import numpy as np
import gymnasium
from raylib import rl, colors
from cffi import FFI
import pyray
import time

class Duelist(pettingzoo.ParallelEnv):
    '''Pufferlib Duelist environment

    Two agents on a 32x32 grid with health values and movement capabilities.
    
    Observation space: Dict with 'position' (x,y float coordinates) and 'health' (current health value)
    Action space: Discrete(5) for movement in cardinal directions (up, down, left, right) and standing still
    
    Health values: 0 (dead), 0.5, 1, 1.5, 2, 2.5, 3
    Movement speed: 0.2 per step
    
    Controls:
    - Move: Arrow Keys (blue agent only)
    - Stand Still: No movement key pressed
    - Swing Sword: S key
    '''
    def __init__(self, render_mode='raylib', keyboard_control=True):
        # Define valid health values
        self.health_values = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        
        # Grid dimensions
        self.grid_size = 32
        self.movement_speed = 0.2
        
        # Initialize agent positions and health
        self.positions = {
            1: np.array([8.0, 8.0], dtype=np.float32),  # Agent 1 starts at (8,8)
            2: np.array([24.0, 24.0], dtype=np.float32)  # Agent 2 starts at (24,24)
        }
        
        self.health = {
            1: 3.0,  # Start with full health
            2: 3.0
        }
        
        # Sword state (0 = not swinging, > 0 = swinging animation frames left)
        self.sword_state = {
            1: 0,
            2: 0
        }
        
        # Sword swing animation duration (in frames)
        self.sword_animation_frames = 5
        
        # Sword hitbox distance (how far the sword extends from the agent)
        self.sword_hitbox_distance = 1.5
        
        # Facing direction (0=up, 1=down, 2=left, 3=right)
        self.facing = {
            1: 0,
            2: 0
        }
        
        # Terminal and truncation flags
        self.terminal = {
            1: False,
            2: False
        }
        
        self.truncated = {
            1: False,
            2: False
        }
        
        self.possible_agents = [1, 2]
        self.agents = [1, 2]
        self.render_mode = render_mode
        self.keyboard_control = keyboard_control
        
        # For backward compatibility
        self.view = np.zeros((2, 5), dtype=np.float32)
        
        # Store actions for keyboard control
        self.keyboard_actions = {
            1: 4,  # Default action (Stand still)
            2: 4   # Default action (Stand still)
        }
        
        # Store sword swing state for keyboard control
        self.keyboard_sword_swing = {
            1: False,
            2: False
        }
        
        # For steps per second calculation
        self.last_time = time.time()
        self.step_count = 0
        self.steps_per_second = 0
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.last_fps_update = time.time()
        
        # Renderer setup
        self.upscale = 16  # Adjust this to change window size
        # Add padding for UI elements
        self.ui_padding = 300  # Padding on the right side for UI (increased from 220)
        self.top_padding = 70  # Padding on the top for UI (increased from 50)
        window_width = self.grid_size * self.upscale + self.ui_padding
        window_height = self.grid_size * self.upscale + self.top_padding
        self.setup_raylib_renderer(window_width, window_height)
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right, 4=stand still
        self.action_mapping = {
            0: np.array([0, -self.movement_speed]),  # Up
            1: np.array([0, self.movement_speed]),   # Down
            2: np.array([-self.movement_speed, 0]),  # Left
            3: np.array([self.movement_speed, 0]),   # Right
            4: np.array([0, 0])                      # Stand Still
        }

    def setup_raylib_renderer(self, width, height):
        """Setup raylib window and rendering components"""
        rl.InitWindow(width, height, "PufferLib Duelist".encode())
        rl.SetTargetFPS(30)  
        
        # Define colors for our grid cells
        self.agent_colors = {
            1: colors.BLUE,   # Agent 1
            2: colors.RED     # Agent 2
        }
        
        # Create texture for rendering
        rendered = np.zeros((height, width, 4), dtype=np.uint8)
        raylib_image = pyray.Image(FFI().from_buffer(rendered.data),
            width, height, 1, pyray.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
        self.texture = rl.LoadTextureFromImage(raylib_image)
        
        # Create rescaler for upscaling the grid
        self.rescaler = np.ones((self.upscale, self.upscale, 1), dtype=np.uint8)

    def _check_keyboard_input(self):
        """Check for keyboard input and update actions accordingly
        Only controls Agent 1 (blue agent) with arrow keys for movement and S for sword"""
        # Agent 1 controls: Arrow keys for movement, S for sword
        # For movement
        if rl.IsKeyDown(rl.KEY_UP):
            self.keyboard_actions[1] = 0  # Up
            self.facing[1] = 0
        elif rl.IsKeyDown(rl.KEY_DOWN):
            self.keyboard_actions[1] = 1  # Down
            self.facing[1] = 1
        elif rl.IsKeyDown(rl.KEY_LEFT):
            self.keyboard_actions[1] = 2  # Left
            self.facing[1] = 2
        elif rl.IsKeyDown(rl.KEY_RIGHT):
            self.keyboard_actions[1] = 3  # Right
            self.facing[1] = 3
        else:
            # No movement key pressed for agent 1 (stand still)
            self.keyboard_actions[1] = 4  # Stand still
        
        # For sword swing - check S key
        if rl.IsKeyPressed(rl.KEY_S):
            self.keyboard_sword_swing[1] = True
        else:
            self.keyboard_sword_swing[1] = False
        
        # Set Agent 2 to always stand still (no keyboard controls)
        self.keyboard_actions[2] = 4  # Stand still
        self.keyboard_sword_swing[2] = False
        
        # Check for reset key - R
        if rl.IsKeyPressed(rl.KEY_R):
            self.reset()

    def observation_space(self, agent):
        return gymnasium.spaces.Dict({
            'position': gymnasium.spaces.Box(
                low=0, 
                high=self.grid_size, 
                shape=(2,), 
                dtype=np.float32
            ),
            'health': gymnasium.spaces.Discrete(len(self.health_values))
        })

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(5)  # 4 movement directions + stand still
    
    def _get_observation(self, agent):
        """Returns the observation for a specific agent"""
        return {
            'position': self.positions[agent].copy(),
            'health': self.health_values.index(self.health[agent])
        }

    def _get_observations(self):
        """Returns observations for all agents"""
        return {agent: self._get_observation(agent) for agent in self.agents}

    def _get_sword_hitbox_position(self, agent):
        """Calculate the position of the sword hitbox based on facing direction"""
        pos = self.positions[agent].copy()
        facing = self.facing[agent]
        
        # Calculate position based on facing direction
        if facing == 0:  # Up
            pos[1] -= self.sword_hitbox_distance
        elif facing == 1:  # Down
            pos[1] += self.sword_hitbox_distance
        elif facing == 2:  # Left
            pos[0] -= self.sword_hitbox_distance
        elif facing == 3:  # Right
            pos[0] += self.sword_hitbox_distance
        
        # Clip to grid boundaries
        return np.clip(pos, 0, self.grid_size-1)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Reset positions
        self.positions[1] = np.array([8.0, 8.0], dtype=np.float32)
        self.positions[2] = np.array([24.0, 24.0], dtype=np.float32)
        
        # Reset health
        self.health[1] = 3.0
        self.health[2] = 3.0
        
        # Reset sword states
        self.sword_state = {1: 0, 2: 0}
        # Make agents face each other initially
        self.facing = {1: 1, 2: 0}  # Agent 1 faces down, Agent 2 faces up
        
        # Reset terminal and truncated flags
        for agent in self.agents:
            self.terminal[agent] = False
            self.truncated[agent] = False
        
        # Reset keyboard actions to stand still
        self.keyboard_actions = {1: 4, 2: 4}
        self.keyboard_sword_swing = {1: False, 2: False}
        
        # For backward compatibility
        self.view = np.zeros((2, 5), dtype=np.float32)
        
        return self._get_observations(), {}

    def step(self, actions):
        # Calculate steps per second
        current_time = time.time()
        self.step_count += 1
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.steps_per_second = self.step_count / (current_time - self.last_fps_update)
            self.step_count = 0
            self.last_fps_update = current_time
            
        # If keyboard control is enabled, use keyboard actions instead
        if self.keyboard_control:
            movement_actions = self.keyboard_actions
            sword_actions = self.keyboard_sword_swing
        else:
            # For programmatic control, we'll use the first bit of the action for movement
            # and check if we have sword swing information
            movement_actions = actions
            sword_actions = {agent: False for agent in self.agents}
            
            # Check if we have sword actions in the info dictionary
            if isinstance(actions, dict) and any('swing_sword' in action for action in actions.values() if isinstance(action, dict)):
                for agent, action_dict in actions.items():
                    if isinstance(action_dict, dict) and 'swing_sword' in action_dict:
                        sword_actions[agent] = action_dict['swing_sword']
                        movement_actions[agent] = action_dict['movement']
        
        # Process each agent's sword action
        for agent, swing in sword_actions.items():
            if swing and self.sword_state[agent] == 0 and not self.terminal[agent]:
                # Start sword swing animation
                self.sword_state[agent] = self.sword_animation_frames
        
        # Process each agent's movement action
        for agent, action in movement_actions.items():
            if not self.terminal[agent]:
                # Apply movement
                movement = self.action_mapping[action]
                self.positions[agent] += movement
                
                # Update facing direction only if the agent is actually moving
                if action != 4:  # If not standing still
                    self.facing[agent] = action
                
                # Clip position to grid boundaries
                self.positions[agent] = np.clip(
                    self.positions[agent], 
                    0, 
                    self.grid_size - 1
                )
        
        # Update sword animation states
        for agent in self.agents:
            if self.sword_state[agent] > 0:
                self.sword_state[agent] -= 1
        
        # Check for sword hits and agent collisions
        self._check_sword_hits()
        self._check_collisions()
        
        # For now, rewards are empty
        rewards = {agent: 0 for agent in self.agents}
        
        # Update info with positions and health
        info = {
            agent: {
                'position': self.positions[agent].copy(),
                'health': self.health[agent],
                'sword_state': self.sword_state[agent],
                'facing': self.facing[agent],
                'score': rewards[agent]  # For backward compatibility
            } for agent in self.agents
        }
        
        return self._get_observations(), rewards, self.terminal, self.truncated, info

    def _check_sword_hits(self):
        """Check if any agent's sword hits another agent"""
        for attacker in self.agents:
            # Only check if the sword is currently swinging
            if self.sword_state[attacker] > 0:
                # Calculate sword hitbox position
                sword_pos = self._get_sword_hitbox_position(attacker)
                
                # Check if sword hits any other agent
                for defender in self.agents:
                    if attacker != defender and not self.terminal[defender]:
                        defender_pos = np.clip(self.positions[defender], 0, self.grid_size-1)
                        
                        # Check if sword hitbox overlaps with defender
                        if np.linalg.norm(sword_pos - defender_pos) <= 1.0:
                            # Sword hit! Reduce defender's health
                            current_idx = self.health_values.index(self.health[defender])
                            if current_idx > 0:
                                self.health[defender] = self.health_values[current_idx - 1]
                            
                            # Check if defender is dead
                            if self.health[defender] == 0:
                                self.terminal[defender] = True

    def _check_collisions(self):
        """Check for collisions between agents and update health"""
        # Get integer positions of agents
        pos1 = np.clip(self.positions[1], 0, self.grid_size-1).astype(int)
        pos2 = np.clip(self.positions[2], 0, self.grid_size-1).astype(int)
        
        # Check if agents are in the same cell (collision)
        if np.array_equal(pos1, pos2):
            # Simple collision logic: both agents lose health
            for agent in self.agents:
                # Find the index of current health in health_values
                current_idx = self.health_values.index(self.health[agent])
                # Reduce health by one level if possible
                if current_idx > 0:
                    self.health[agent] = self.health_values[current_idx - 1]
                
                # Check if agent is dead
                if self.health[agent] == 0:
                    self.terminal[agent] = True

    def render(self):
        # Check for keyboard input if keyboard control is enabled
        if self.keyboard_control:
            self._check_keyboard_input()
        
        # Render the frame using raylib
        rl.BeginDrawing()
        rl.ClearBackground(colors.BLACK)
        
        # Define arena boundaries
        arena_x = 10  # Add a small margin on the left
        arena_y = self.top_padding
        arena_width = self.grid_size * self.upscale
        arena_height = self.grid_size * self.upscale
        
        # Draw a border around the arena
        border_thickness = 2
        rl.DrawRectangleLinesEx(
            (arena_x, arena_y, arena_width, arena_height),
            border_thickness,
            colors.WHITE
        )
        
        # Draw a grid for reference inside the arena
        grid_color = colors.DARKGRAY
        for x in range(0, self.grid_size + 1):
            # Draw vertical lines
            start_x = arena_x + x * self.upscale
            rl.DrawLine(start_x, arena_y, start_x, arena_y + arena_height, grid_color)
            
            # Draw horizontal lines
            start_y = arena_y + x * self.upscale
            rl.DrawLine(arena_x, start_y, arena_x + arena_width, start_y, grid_color)
        
        # Draw agents as triangles based on their facing direction
        for agent in self.agents:
            pos = self.positions[agent]
            x, y = pos
            
            # Convert to screen coordinates with arena offset
            x_scaled = int(arena_x + x * self.upscale + self.upscale / 2)
            y_scaled = int(arena_y + y * self.upscale + self.upscale / 2)
            
            # Calculate triangle size
            triangle_size = self.upscale * 0.8  # Slightly smaller than a full cell
            
            # Choose color based on agent - use pure primary colors for better visibility
            if agent == 1:
                # Pure saturated blue - not using the dictionary to ensure color consistency
                agent_color = pyray.Color(0, 64, 255, 255)  # Brighter blue with full alpha
                outline_color = colors.WHITE
            else:
                # Pure saturated red - not using the dictionary to ensure color consistency
                agent_color = pyray.Color(255, 40, 40, 255)  # Brighter red with full alpha
                outline_color = colors.WHITE
            
            # Define triangle points based on facing direction
            if self.facing[agent] == 0:  # Up
                p1 = (x_scaled, y_scaled - triangle_size/2)  # Point
                p2 = (x_scaled - triangle_size/2, y_scaled + triangle_size/2)  # Left corner
                p3 = (x_scaled + triangle_size/2, y_scaled + triangle_size/2)  # Right corner
            elif self.facing[agent] == 1:  # Down
                p1 = (x_scaled, y_scaled + triangle_size/2)  # Point
                p2 = (x_scaled - triangle_size/2, y_scaled - triangle_size/2)  # Left corner
                p3 = (x_scaled + triangle_size/2, y_scaled - triangle_size/2)  # Right corner
            elif self.facing[agent] == 2:  # Left
                p1 = (x_scaled - triangle_size/2, y_scaled)  # Point
                p2 = (x_scaled + triangle_size/2, y_scaled - triangle_size/2)  # Top corner
                p3 = (x_scaled + triangle_size/2, y_scaled + triangle_size/2)  # Bottom corner
            elif self.facing[agent] == 3:  # Right
                p1 = (x_scaled + triangle_size/2, y_scaled)  # Point
                p2 = (x_scaled - triangle_size/2, y_scaled - triangle_size/2)  # Top corner
                p3 = (x_scaled - triangle_size/2, y_scaled + triangle_size/2)  # Bottom corner
            
            # Draw the filled triangle with full opacity
            rl.DrawTriangle(p1, p2, p3, agent_color)
            
            # Draw a thick outline for better visibility
            rl.DrawLineEx(p1, p2, 2.0, outline_color)
            rl.DrawLineEx(p2, p3, 2.0, outline_color)
            rl.DrawLineEx(p3, p1, 2.0, outline_color)
        
        # Draw sword hitboxes if swords are swinging
        for agent in self.agents:
            if self.sword_state[agent] > 0:
                sword_pos = self._get_sword_hitbox_position(agent)
                sx, sy = sword_pos
                
                # Convert to screen coordinates
                sx_scaled = int(arena_x + sx * self.upscale + self.upscale / 2)
                sy_scaled = int(arena_y + sy * self.upscale + self.upscale / 2)
                
                # Draw sword hitbox with color matching agent and white outline
                sword_size = self.upscale * 0.5
                
                # Use the same colors as the agent triangles for consistency
                if agent == 1:
                    sword_color = pyray.Color(0, 64, 255, 200)  # Blue with some transparency
                else:
                    sword_color = pyray.Color(255, 40, 40, 200)  # Red with some transparency
                
                rl.DrawCircle(sx_scaled, sy_scaled, sword_size, sword_color)
                rl.DrawCircleLines(sx_scaled, sy_scaled, sword_size, colors.WHITE)
        
        # Header area (top of screen)
        rl.DrawText("PufferLib Duelist".encode(), 20, 20, 30, colors.WHITE)
        rl.DrawText(f"Steps/sec: {self.steps_per_second:.1f}".encode(), 320, 25, 20, colors.LIME)
        
        # Draw UI elements (right side of arena)
        ui_x = arena_x + arena_width + 20
        
        # Draw UI background for better readability
        ui_bg_width = self.ui_padding - 40  # Increased width for sidebar
        ui_bg_height = arena_height
        rl.DrawRectangle(ui_x - 10, arena_y, ui_bg_width, ui_bg_height, colors.BLACK)
        rl.DrawRectangleLinesEx(
            (ui_x - 10, arena_y, ui_bg_width, ui_bg_height),
            1,
            colors.DARKGRAY
        )
        
        # Title for the sidebar
        rl.DrawText("Game Information".encode(), ui_x, arena_y + 10, 24, colors.WHITE)
        rl.DrawLine(ui_x, arena_y + 40, ui_x + ui_bg_width - 20, arena_y + 40, colors.DARKGRAY)
        
        # Display health information for both agents
        health_y = arena_y + 60
        rl.DrawText("Health:".encode(), ui_x, health_y, 20, colors.WHITE)
        rl.DrawText(f"Agent 1: {self.health[1]}".encode(), ui_x + 20, health_y + 25, 18, colors.BLUE)
        rl.DrawText(f"Agent 2: {self.health[2]}".encode(), ui_x + 20, health_y + 50, 18, colors.RED)
        
        # Display agent positions for debugging
        position_y = health_y + 90
        rl.DrawText("Position:".encode(), ui_x, position_y, 20, colors.WHITE)
        rl.DrawText(f"Agent 1: ({self.positions[1][0]:.1f}, {self.positions[1][1]:.1f})".encode(), 
                   ui_x + 20, position_y + 25, 18, colors.BLUE)
        rl.DrawText(f"Agent 2: ({self.positions[2][0]:.1f}, {self.positions[2][1]:.1f})".encode(), 
                   ui_x + 20, position_y + 50, 18, colors.RED)
        
        # Display facing directions
        facing_y = position_y + 90
        facing_names = ["Up", "Down", "Left", "Right"]
        rl.DrawText("Facing Direction:".encode(), ui_x, facing_y, 20, colors.WHITE)
        rl.DrawText(f"Agent 1: {facing_names[self.facing[1]]}".encode(), 
                   ui_x + 20, facing_y + 25, 18, colors.BLUE)
        rl.DrawText(f"Agent 2: {facing_names[self.facing[2]]}".encode(), 
                   ui_x + 20, facing_y + 50, 18, colors.RED)
        
        # Display sword states
        sword_y = facing_y + 90
        rl.DrawText("Sword State:".encode(), ui_x, sword_y, 20, colors.WHITE)
        rl.DrawText(f"Agent 1: {'Swinging' if self.sword_state[1] > 0 else 'Ready'}".encode(), 
                   ui_x + 20, sword_y + 25, 18, colors.BLUE)
        rl.DrawText(f"Agent 2: {'Swinging' if self.sword_state[2] > 0 else 'Ready'}".encode(), 
                   ui_x + 20, sword_y + 50, 18, colors.RED)
        
        # Display controls if keyboard control is enabled
        if self.keyboard_control:
            controls_y = sword_y + 90
            rl.DrawLine(ui_x, controls_y, ui_x + ui_bg_width - 20, controls_y, colors.DARKGRAY)
            rl.DrawText("Controls:".encode(), ui_x, controls_y + 20, 20, colors.WHITE)
            rl.DrawText("Blue Agent (Arrow Keys)".encode(), ui_x + 20, controls_y + 50, 18, colors.BLUE)
            rl.DrawText("Sword Swing: S key".encode(), ui_x + 20, controls_y + 75, 18, colors.WHITE)
            rl.DrawText("Reset Game: R".encode(), ui_x + 20, controls_y + 100, 18, colors.WHITE)
            rl.DrawText("Exit Game: ESC".encode(), ui_x + 20, controls_y + 125, 18, colors.WHITE)
        
        # Handle exit conditions
        if rl.IsKeyPressed(rl.KEY_ESCAPE):
            exit(0)
            
        rl.EndDrawing()
        
        # Return a screenshot of the window (similar to Atari)
        def cdata_to_numpy():
            image = rl.LoadImageFromScreen()
            data_pointer = image.data
            width = image.width
            height = image.height
            channels = 4
            data_size = width * height * channels
            cdata = FFI().buffer(data_pointer, data_size)
            return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))
            
        return cdata_to_numpy()
        
    def _update_grid(self):
        """This method is kept for backward compatibility but no longer updates a grid"""
        pass