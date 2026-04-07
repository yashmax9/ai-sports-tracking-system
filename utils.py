def compute_iou(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    xi1 = max(x1,x2)
    yi1 = max(y1,y2)
    xi2 = min(x1+w1,x2+w2)
    yi2 = min(y1+h1,y2+h2)

    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter

    return inter/union if union > 0 else 0