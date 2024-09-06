import pygame
from Config import PhysicsSettings, GameSettings, MapSettings, PlayerSettings
from Vector import Vector

# --- Physics class ---
class Physics:
    def __init__(self):
        super().__init__()
        self.force_applied = Vector(0, 0)

    def apply_force(self, force, moveable_object):
        acceleration = force / moveable_object.mass
        moveable_object.velocity += acceleration

    def apply_friction(self, moveable_object, dt_tick):
        friction_force = PhysicsSettings.GROUND_COEF_FRIC if moveable_object.on_ground else PhysicsSettings.AIR_COEF_FRIC
        friction = moveable_object.velocity.normalize() * (-friction_force * 100)
        # Scale friction by delta time and ensure it doesn't reverse the velocity
        friction_to_apply = friction * dt_tick
        # If the friction force is greater than the current velocity, clamp it to zero
        if friction_to_apply.magnitude() > moveable_object.velocity.magnitude():
            moveable_object.velocity = Vector(0, 0)
        else:
            moveable_object.velocity += friction_to_apply
            
    def apply_gravity(self, moveable_object, dt_tick):
        if not moveable_object.on_ground:
            gravity = Vector.calculate_force_vector(90, PhysicsSettings.GRAVITY_ACC)
            gravity_force = gravity * dt_tick
            resulting_velocity = moveable_object.velocity + gravity_force / moveable_object.mass
            terminal_velocity = PhysicsSettings.TERMINAL_VELOCITY
            if resulting_velocity.magnitude() > terminal_velocity:
                resulting_velocity = resulting_velocity.normalize() * terminal_velocity
            moveable_object.velocity = resulting_velocity
        if self.force_applied.magnitude() > 0:
            self.apply_force(self.force_applied, moveable_object)
            self.force_applied = Vector(0, 0)

    def check_boundaries(self, moveable_object):
        # Extract common values for easier access
        obj_width = moveable_object.rect.width
        obj_height = moveable_object.rect.height
        screen_width = GameSettings.SCREEN_WIDTH
        screen_height = GameSettings.SCREEN_HEIGHT
        
        # X-axis boundary check (left and right)
        if moveable_object.position.x < 0:
            moveable_object.position.x = 0
            moveable_object.velocity.x = 0
        elif moveable_object.position.x + obj_width > screen_width:
            moveable_object.position.x = screen_width - obj_width
            moveable_object.velocity.x = 0
        
        # Y-axis boundary check (top and bottom)
        if moveable_object.position.y < 0:
            moveable_object.position.y = 0
            moveable_object.velocity.y = 0
        elif moveable_object.position.y + obj_height > screen_height:
            moveable_object.position.y = screen_height - obj_height
            moveable_object.velocity.y = 0
        
        # Ground check (bottom of the screen)
        if moveable_object.position.y >= screen_height - obj_height - 0.1:
            moveable_object.on_ground = True
        else:
            moveable_object.on_ground = False

    def check_collisions():
        collisions = pygame.sprite.groupcollide(PlayerSettings.PLAYER_SPRITE, MapSettings.MAP_SPRITE, False, False)
        
        for player_sprite, map_sprites in collisions.items():
            for map_sprite in map_sprites:
                # Calculate collision vectors
                player_pos = pygame.Vector2(player_sprite.rect.center)
                map_pos = pygame.Vector2(map_sprite.rect.center)
                direction_vector = player_pos - map_pos

                # Check if direction_vector length is greater than zero to avoid division by zero
                if direction_vector.length() == 0:
                    continue     

    def update(self, moveable_object, dt_tick):
        self.apply_gravity(moveable_object, dt_tick)
        self.apply_friction(moveable_object, dt_tick)
        self.check_boundaries(moveable_object)
        Physics.check_collisions()
