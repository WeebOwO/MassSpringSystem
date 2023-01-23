import taichi as ti
from engine import ParticleEngine

ti.init(arch=ti.cpu)

gui = ti.GUI("Spring Mass System", res=(512, 512), background_color=0xdddddd)
engine = ParticleEngine(max_particle_num=1024, particle_mass=1.0, young_modulus=1000, delta_time=5 * 1e-3)

def process_event(): 
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == ti.GUI.LMB:
            engine.add_new_particle(e.pos[0], e.pos[1], int(gui.is_pressed(ti.GUI.SHIFT)))
        elif e.key == 'f':
            engine.set_dashpot(True)
    return

def render():
    n = engine.current_num_particle[None]
    for i in range(n):
        fixed = engine.particle_fixed[i]
        color = 0xff0000 if fixed else 0x111111
        gui.circle(pos=engine.particle_pos[i], color=color, radius=5)

    for i in range(n):
        for j in range(i + 1, n): 
            if engine.rest_length[i, j] != 0:
                begin = engine.particle_pos[i]
                end = engine.particle_pos[j]
                gui.line(begin, end, radius=2, color=0x444444)
    
    gui.text(content=
            f'Left click: add mass point (with shift to fix)',
            pos=(0, 0.99),
            color=0x0)
    gui.show()

def main():
    # main loop
    while True:
        process_event()
        engine.step()
        render()

if __name__ == "__main__":
    main()
