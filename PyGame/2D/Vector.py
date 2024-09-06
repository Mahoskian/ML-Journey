import math

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)

    def __repr__(self):
        return f"Vector({self.x:.2f}, {self.y:.2f})"
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def copy(self):
        return Vector(self.x, self.y)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector()
        return self / mag
    
    def reflect(self, normal):
        dot_product = 2 * self.dot(normal)
        reflection = Vector(self.x - dot_product * normal.x, self.y - dot_product * normal.y)
        return reflection

    def calculate_force_vector(angle_degrees, magnitude):
        angle_radians = math.radians(angle_degrees)
        return Vector(
            magnitude * math.cos(angle_radians),
            magnitude * math.sin(angle_radians)
        )
        
    def angle(self):
        return math.degrees(math.atan2(self.y, self.x))  # Angle in degrees