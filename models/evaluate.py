def evaluate(model, dataset):
    a = dataset.data["wiki_intro"].apply(lambda x: (x, 1))
    b = dataset.data["generated_intro"].apply(lambda x: (x, 0))
    data = list(a)
    data.extend(list(b))
    rightcount = 0
    errorcount = 0
    for item in data:
        try:
            if model(item[0]) == item[1]:
                rightcount += 1
        except Exception:
            errorcount += 1
            continue
    return rightcount / (2 * len(data) - errorcount)
