import pygame
from Physics import Physics
from Vector import Vector
from Config import BulletSettings

class Bullet(pygame.sprite.Sprite):
    def __init__(self, position, direction):
        super().__init__()
        self.name = "Bullet"
        self.position = position
        self.velocity = Vector(0, 0)
        self.mass = BulletSettings.MASS
        self.on_ground = False
        
        self.time_alive = 0
        
        self.image = pygame.Surface((BulletSettings.RADIUS * 2, BulletSettings.RADIUS * 2), pygame.SRCALPHA)  # Create a surface with transparency
        pygame.draw.circle(self.image, pygame.Color('purple'), (BulletSettings.RADIUS, BulletSettings.RADIUS), BulletSettings.RADIUS)  # Draw a purple circle
        self.rect = self.image.get_rect(center=(self.position.x, self.position.y))

        self.bullet_physics = Physics()
        
        # Apply initial force to the bullet
        force_vector = Vector.calculate_force_vector(direction, BulletSettings.FORCE)
        self.bullet_physics.apply_force(force_vector, self)

    def update(self, dt_tick):
        self.time_alive += dt_tick * 1000  # Convert dt_tick to milliseconds
        if self.time_alive > BulletSettings.LIFE_TIME:
            self.kill()  # Remove from all groups
        # Update the bullet's physics and position
        self.bullet_physics.update(self, dt_tick)
        self.position += self.velocity * dt_tick
        self.rect.center = (self.position.x, self.position.y)
