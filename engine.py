import taichi as ti

@ti.data_oriented
class ParticleEngine():
    def __init__(self, max_particle_num, particle_mass, young_modulus, delta_time) -> None:

        self.young_modulus = ti.field(dtype=ti.f32, shape=())
        self.young_modulus[None] = young_modulus

        self.max_particles_num = max_particle_num
        self.particle_mass = particle_mass
        self.current_num_particle = ti.field(dtype=ti.i32, shape=())
        
        self.particle_pos = ti.Vector.field(2, dtype=ti.f32, shape=self.max_particles_num)
        self.particle_velocity = ti.Vector.field(2, dtype=ti.f32, shape=self.max_particles_num)
        self.particle_force = ti.Vector.field(2, dtype=ti.f32, shape=self.max_particles_num)
        self.particle_fixed = ti.field(dtype=ti.i32, shape=self.max_particles_num)

        self.rest_length = ti.field(dtype=ti.f32, shape=(self.max_particles_num, self.max_particles_num))
        self.dashpot_damping = ti.field(dtype=ti.f32, shape=())
        self.dashpot_damping[None] = 100

        self.add_new_particle(0.3, 0.3, False)
        self.add_new_particle(0.3, 0.4, False)
        self.add_new_particle(0.4, 0.4, False)
        self.add_new_particle(0.5, 0.7, True)

        self.delta_time = delta_time
        self.enable_dashpot = False

    def set_dashpot(self, state):
        self.enable_dashpot = state

    @ti.kernel
    def add_new_particle(self, pos_x : ti.f32, pos_y : ti.f32, fixed : ti.i32):
        new_particle_id = self.current_num_particle[None]
        self.particle_pos[new_particle_id] = [pos_x, pos_y]
        self.particle_velocity[new_particle_id] = [0.0, 0.0]
        self.particle_fixed[new_particle_id] = fixed
        
        self.current_num_particle[None] += 1

        # check connnected state
        for i in range(new_particle_id):
            distance = (self.particle_pos[new_particle_id] - self.particle_pos[i]).norm()
            if distance < 0.15:
                self.rest_length[i, new_particle_id] = 0.1
                self.rest_length[new_particle_id, i] = 0.1

    @ti.kernel
    def step(self):
        particle_num = self.current_num_particle[None]
        gravity = 9.8
        # calculate force 
        for i in range(particle_num):
            self.particle_force[i] = ti.Vector([0.0, -gravity]) * self.particle_mass
            for j in range(particle_num):
                if self.rest_length[i, j] != 0:
                    # spring force i 
                    vec_ij = self.particle_pos[i] - self.particle_pos[j]
                    direction = vec_ij.normalized()
                    self.particle_force[i] += -self.young_modulus[None] * (vec_ij.norm() / self.rest_length[i, j] - 1) * direction
        # solve 
        for i in range(particle_num):
            if not self.particle_fixed[i]:
                self.particle_velocity[i] += self.delta_time * self.particle_force[i] / self.particle_mass
                self.particle_velocity[i] *= ti.exp(-self.delta_time)
                self.particle_pos[i] += self.particle_velocity[i] *self.delta_time

            else:
                self.particle_velocity[i] = ti.Vector([0.0, 0.0])
            
            # collison part 
            for d in ti.static(range(2)):
                if self.particle_pos[i][d] < 0:
                    self.particle_pos[i][d] = 0
                    self.particle_velocity[i][d] = 0

                if self.particle_pos[i][d] > 1:
                    self.particle_pos[i][d] = 0
                    self.particle_velocity[i][d] = 0
            
