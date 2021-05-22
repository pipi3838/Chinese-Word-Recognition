store = defaultdict(list)
found = 0
not_found = 0
null_cnt = 0
total = 0

for file in tqdm(glob(os.path.join(path, 'Origin', '*.jpg'))):
    img_name = file.split('/')[-1]
    store[img_name[-5]].append(img_name.split('_')[0])
    ori_img = cv2.imread(file, 1)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    total += 1
    if ori_img is None:
#         print('NULL image ', img_name)
        null_cnt += 1
        continue

    _, thre_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
#     im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,2)

    thre_img = cv2.fastNlMeansDenoising(thre_img, h=13, searchWindowSize=7)
    kernel = np.ones((3,3), np.uint8)
    thre_img = cv2.erode(thre_img, kernel, iterations = 1)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dilate = cv2.dilate(thre_img, rect_kernel, iterations = 1)
    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        bound = cv2.resize(img, (50,50))
        not_found += 1
    else:
        max_area = -1
        bx, by, bw, bh = None, None, None, None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > max_area:
                bx, by, bw, bh = x, y, w, h
                max_area = w * h
            # Drawing a rectangle on copied image
        bound = cv2.resize(img[by:by+bh, bx:bx+bw], (50, 50))
        rect = cv2.rectangle(ori_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
        found += 1
    save_path = os.path.join(path, 'Modified', img_name)
    cv2.imwrite(save_path, bound)
    
print('Total {} img found {} bounding box\t {} not found\t{} null images'.format(total, found, not_found, null_cnt))
        #     plt.imshow(img, cmap='gray', vmin=0, vmax=255)
#     imgs = [thre_img, dilate, ori_img, bound]
#     titles = ['thres', 'dilate', 'result', 'bounding']
#     fig, axs = plt.subplots(nrows=1, ncols=4)
#     for i, ax in enumerate(axs.flatten()):
#         plt.sca(ax)
#         if i != 2: plt.imshow(imgs[i], 'gray', vmin=0, vmax=255)
#         else: plt.imshow(imgs[i])
#         plt.title('Image: {}'.format(titles[i]))

#     #plt.tight_layout()
#     plt.show()
#     input()

# print('Total Image Cnt ', cnt)
# print('Total Character ', len(store))