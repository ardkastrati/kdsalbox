def base64_to_png(base64str, output_filepath):
    import base64
    imgdata = base64.b64decode(base64str)
    filename = output_filepath
    with open(filename, 'wb') as f:
        f.write(imgdata)


img = input()
base64_to_png(img, "test.png")
