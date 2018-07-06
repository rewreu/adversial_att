import matplotlib.pyplot as plt
def show(original, perturbed):
    # plot the orginal image and attacked image
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    orr_img = original[0]
    orr_img = orr_img.transpose(0,2).transpose(0,1).cpu().detach().numpy()
    plt.imshow(orr_img)
    out_img = perturbed[0]
    out_img = out_img.transpose(0,2).transpose(0,1).cpu().detach().numpy()
    plt.subplot(122)
    plt.imshow(out_img)