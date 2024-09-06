import pygame
import random
from Physics import Physics
from Config import GameSettings, PlayerSettings, BulletSettings
from Vector import Vector
from Bullet import Bullet

# --- Player class ---
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.name = "Player"
        self.position = Vector(random.randint(100, 700), GameSettings.SCREEN_HEIGHT - 100)
        self.velocity = Vector(0, 0)
        self.mass = PlayerSettings.MASS
        self.on_ground = False
        
        self.last_shot_time = 0
        
        self.image = pygame.Surface(PlayerSettings.SIZE)
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect(center=(self.position.x, self.position.y))

        self.player_physics = Physics()

    def update(self, dt_tick):
        self.handle_input()
        self.player_physics.update(self, dt_tick)
        self.update_position(dt_tick)
        self.rect.topleft = (self.position.x, self.position.y)
        BulletSettings.BULLET_SPRITE.update(dt_tick)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.player_physics.force_applied += Vector.calculate_force_vector(-90, PlayerSettings.W_FORCE)  # Example direction
        if keys[pygame.K_a]:
            self.player_physics.force_applied += Vector.calculate_force_vector(180, PlayerSettings.A_FORCE)
        if keys[pygame.K_s]:
            self.player_physics.force_applied += Vector.calculate_force_vector(90, PlayerSettings.S_FORCE)  # Example direction
        if keys[pygame.K_d]:
            self.player_physics.force_applied += Vector.calculate_force_vector(0, PlayerSettings.D_FORCE)
        if keys[pygame.K_SPACE] and self.on_ground:
            self.player_physics.force_applied += Vector.calculate_force_vector(-90, PlayerSettings.SPACE_FORCE)
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            self.shoot()

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time >= PlayerSettings.SHOOT_COOLDOWN:
            mouse_position = pygame.mouse.get_pos()
            mouse_vector = Vector(mouse_position[0], mouse_position[1])
            direction_vector = mouse_vector - self.position
            direction_angle = direction_vector.angle()
            bullet = Bullet(self.position.copy(), direction_angle)
            BulletSettings.BULLET_SPRITE.add(bullet)
            self.last_shot_time = current_time
    
    def update_position(self, dt_tick):
        self.position += self.velocity * dt_tick

    def bounce(self, normal_vector):
        if self.on_ground:
            self.velocity = self.velocity.reflect(normal_vector)
