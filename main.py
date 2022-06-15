

from model.gait_recog import Discriminator

def main():

    discriminator = Discriminator.model(128, 128, 1)
    

if __name__ == '__main__':
    main()