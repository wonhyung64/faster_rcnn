
#%%

batch_size = 2
hyper_params['batch_size'] = batch_size

NMS = RoIBBox(anchors, hyper_params, test=True, name='roi_bboxes')
Pooling = RoIPooling(hyper_params, name="roi_pooling")
decode = Decoder(hyper_params)

test_dir = r"C:\won\data\pascal_voc\voc2007_np\test\\"

for attempt in range(attempts):

    res_filename = [test_filename[i] for i in range(attempt*batch_size, attempt*batch_size + batch_size)]
    batch_data = np.array([np.load(test_dir + test_filename[i] + ".npy", allow_pickle=True) for i in range(attempt*batch_size, attempt*batch_size+batch_size)])

    img, gt_boxes, gt_labels = preprocessing_utils.preprocessing(batch_data, hyper_params["batch_size"], hyper_params["img_size"], hyper_params["img_size"], evaluate=True)
    
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model.predict(img)
    roi_bboxes, roi_scores = NMS([rpn_reg_output, rpn_cls_output, gt_labels])
    pooled_roi = Pooling([feature_map, roi_bboxes])
    pred_deltas, pred_label_probs = frcnn_model.predict(pooled_roi)
    final_bboxes, final_labels, final_scores = decode([roi_bboxes, pred_deltas, pred_label_probs])

    img_size = img.shape[1]
    