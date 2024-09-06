import pygame
from Map import Wall, BouncyBlock, StickyBlock
from Config import PhysicsSettings, MapSettings, PlayerSettings

# --- Physics class ---
class Physics:
    def __init__(self):
        self.force_applied = pygame.Vector2(0, 0)

    def apply_force(self, force, moveable_object):
        acceleration = force / moveable_object.mass
        moveable_object.velocity += acceleration

    def apply_friction(self, moveable_object, dt_tick):
        friction_force = PhysicsSettings.GROUND_COEF_FRIC if moveable_object.on_ground else PhysicsSettings.AIR_COEF_FRIC
        friction = moveable_object.velocity.normalize() * (-friction_force * 100)
        friction_to_apply = friction * dt_tick
        if friction_to_apply.length() > moveable_object.velocity.length():
            moveable_object.velocity = pygame.Vector2(0, 0)
        else:
            moveable_object.velocity += friction_to_apply

    def apply_gravity(self, moveable_object, dt_tick):
        if not moveable_object.on_ground:
            gravity = pygame.Vector2(0, PhysicsSettings.GRAVITY_ACC)
            gravity_force = gravity * dt_tick
            resulting_velocity = moveable_object.velocity + gravity_force / moveable_object.mass
            terminal_velocity = PhysicsSettings.TERMINAL_VELOCITY
            if resulting_velocity.length() > terminal_velocity:
                resulting_velocity = resulting_velocity.normalize() * terminal_velocity
            moveable_object.velocity = resulting_velocity
        
        if self.force_applied.length() > 0:
            self.apply_force(self.force_applied, moveable_object)
            self.force_applied = pygame.Vector2(0, 0)

    def check_wall_collisions(self):
        collisions = pygame.sprite.groupcollide(PlayerSettings.PLAYER_SPRITE, MapSettings.MAP_SPRITE, False, False)
        
        for player_sprite, map_sprites in collisions.items():
            player_pos = pygame.Vector2(player_sprite.rect.center)
            for map_sprite in map_sprites:
                map_pos = pygame.Vector2(map_sprite.rect.center)
                direction_vector = player_pos - map_pos

                if direction_vector.length() == 0:
                    continue
                
                collision_normal = direction_vector.normalize()
                player_velocity = pygame.Vector2(PlayerSettings.PLAYER_OBJ.velocity.x, PlayerSettings.PLAYER_OBJ.velocity.y)
                velocity_normal = player_velocity.project(collision_normal)
                velocity_tangential = player_velocity - velocity_normal
                
                if isinstance(map_sprite, BouncyBlock):
                    PlayerSettings.PLAYER_OBJ.velocity += -2 * velocity_normal
                elif isinstance(map_sprite, StickyBlock):
                    PlayerSettings.PLAYER_OBJ.velocity = pygame.Vector2(0, 0)
                else:
                    PlayerSettings.PLAYER_OBJ.velocity -= velocity_normal

    def update(self, moveable_object, dt_tick):
        self.apply_gravity(moveable_object, dt_tick)
        self.apply_friction(moveable_object, dt_tick)
        self.check_wall_collisions()
