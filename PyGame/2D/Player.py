import pygame
import random
from Physics import Physics
from Config import GameSettings, PlayerSettings, BulletSettings
from Bullet import Bullet

# --- Player class ---
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.name = "Player"
        self.position = pygame.Vector2(random.randint(100, 700), GameSettings.SCREEN_HEIGHT - 100)
        self.velocity = pygame.Vector2(0, 0)
        self.mass = PlayerSettings.MASS
        self.on_ground = False
        self.last_shot_time = 0

        self.image = pygame.Surface(PlayerSettings.SIZE)
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect(center=self.position)

        self.player_physics = Physics()

    def update(self, dt_tick):
        self.handle_input()
        self.player_physics.update(self, dt_tick)
        self.update_position(dt_tick)
        self.rect.topleft = self.position
        BulletSettings.BULLET_SPRITE.update(dt_tick)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        forces = {
            pygame.K_w: pygame.Vector2(0, -PlayerSettings.W_FORCE),
            pygame.K_a: pygame.Vector2(-PlayerSettings.A_FORCE, 0),
            pygame.K_s: pygame.Vector2(0, PlayerSettings.S_FORCE),
            pygame.K_d: pygame.Vector2(PlayerSettings.D_FORCE, 0),
        }
        
        for key, force in forces.items():
            if keys[key]:
                self.player_physics.force_applied += force
        
        if keys[pygame.K_SPACE] and self.on_ground:
            self.player_physics.force_applied += pygame.Vector2(0, -PlayerSettings.SPACE_FORCE)
        
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            self.shoot()

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time >= PlayerSettings.SHOOT_COOLDOWN:
            mouse_position = pygame.Vector2(pygame.mouse.get_pos())
            direction_vector = mouse_position - self.position
            direction_angle = direction_vector.angle_to(pygame.Vector2(1, 0))
            bullet = Bullet(self.position.copy(), direction_angle)
            BulletSettings.BULLET_SPRITE.add(bullet)
            self.last_shot_time = current_time
    
    def update_position(self, dt_tick):
        self.position += self.velocity * dt_tick

    def bounce(self, normal_vector):
        if self.on_ground:
            self.velocity = self.velocity.reflect(normal_vector)
