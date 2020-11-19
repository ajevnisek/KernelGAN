import matplotlib.pyplot as plt
from scipy.io import loadmat

# full_image_original = loadmat("sandbox/interesting_result/kernel_diffs/frame25/all_image including_face/original_frame25/original_frame25_kernel_x2.mat")
# full_image_fake = loadmat("sandbox/interesting_result/kernel_diffs/frame25/all_image including_face/fake_frame25/fake_frame25_kernel_x2.mat")
# context_fake = loadmat("sandbox/interesting_result/kernel_diffs/frame25/image_with_face_cropped/fake_frame25/fake_frame25_kernel_x2.mat")
# context_original = loadmat("sandbox/interesting_result/kernel_diffs/frame25/image_with_face_cropped/original_frame25/original_frame25_kernel_x2.mat")

full_image_original = loadmat(
    "sandbox/interesting_result/kernel_diffs/frame268/all_image"
    "/original_frame268/original_frame268_kernel_x2.mat")
full_image_fake = loadmat("sandbox/interesting_result/kernel_diffs/frame268"
                          "/all_image/fake_frame268/fake_frame268_kernel_x2.mat")
context_fake = loadmat("sandbox/interesting_result/kernel_diffs/frame268"
                       "/image_with_face_cropped/fake_frame268/fake_frame268_kernel_x2.mat")
context_original = loadmat("sandbox/interesting_result/kernel_diffs/frame268"
                           "/image_with_face_cropped/original_frame268"
                           "/original_frame268_kernel_x2.mat")

full_image_original = full_image_original['Kernel']
full_image_fake = full_image_fake['Kernel']
context_fake = context_fake['Kernel']
context_original = context_original['Kernel']

ssd_real = sum(sum((full_image_original - context_original)**2))
ssd_fake = sum(sum((full_image_fake - context_fake)**2))
print(f"ssd for real image: "
      f"{ssd_real}")
print(f"ssd for fake image: "
      f"{ssd_fake}")

fig, axs = plt.subplots(2,2)
fig.figsize=(17,17)
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]

ax1.imshow(full_image_original, extent=[1,17,1,17])
ax1.set_title('Real image, kernel est on full image')
ax2.imshow(context_original, extent=[1,17,1,17])
ax2.set_title('Real image, kernel est on context')
ax3.imshow(full_image_fake, extent=[1,17,1,17])
ax3.set_title('Fake image, kernel est on full image')
ax4.imshow(context_fake, extent=[1,17,1,17])
ax4.set_title('Fake image, kernel est on context')

plt.tight_layout()
# plt.suptitle(f"ssd for real image: {ssd_real},\n ssd fake: {ssd_fake}")
# plt.savefig("sandbox/interesting_result/kernel_diffs/frame25/all_kernels.png")
plt.savefig("sandbox/interesting_result/kernel_diffs/frame268/all_kernels.png")
plt.show()
