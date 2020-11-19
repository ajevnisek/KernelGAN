cropped_real = cv2.imread("sandbox/interesting_result/frame268/image_with_face_cropped/original_frame268/ZSSR_original_frame268.png")
cropped_fake = cv2.imread("sandbox/interesting_result/frame268/image_with_face_cropped/fake_frame268/ZSSR_fake_frame268.png")
all_real = cv2.imread("results/original_frame268/ZSSR_original_frame268.png")
all_fake = cv2.imread("sandbox/interesting_result/frame268/all_image/fake_frame268/ZSSR_fake_frame268.png")
diff_real = all_real - cropped_real
diff_fake = all_fake - cropped_fake
diff_real_gray = cv2.cvtColor(diff_real, cv2.COLOR_BGR2GRAY)
diff_fake_gray = cv2.cvtColor(diff_fake, cv2.COLOR_BGR2GRAY)
cv2.imwrite("sandbox/interesting_result/frame268/diff_real.png", diff_real_gray)
cv2.imwrite("sandbox/interesting_result/frame268/diff_fake.png", diff_fake_gray)


