import imageio

img = imageio.imread("./results/d2sn/test_latest/images/UV_diffuse_merged_synthesized_image.jpg")
specular = img[:,:,0]

imageio.imwrite("./results/d2sn/test_latest/images/specular.png", specular)