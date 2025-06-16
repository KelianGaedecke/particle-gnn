from src.model import ParticleGNN
from src.data_utils import generate_particles

def main():
    model = ParticleGNN()
    data = generate_particles(20)
    delta = model(data.x, data.pos, data.batch)
    next_pos = data.pos + delta
    print("Next position:\n", next_pos)

if __name__ == '__main__':
    main()