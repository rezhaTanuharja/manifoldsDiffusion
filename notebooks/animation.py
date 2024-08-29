import imageio


with imageio.get_writer('downloads/post_animation.gif', mode='I', duration=0.1, loop = 1) as writer:
    for i in range(100):
        filename = f'downloads/noising/human_{i}.png'
        image = imageio.imread(filename)
        writer.append_data(image)

print('finished')
