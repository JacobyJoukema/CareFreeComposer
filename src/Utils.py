def compileCompositions(directory):
    out = []
    for phial in os.listdir(directory):
        if phial.filename.endswith(".mid"):
            out.append(phial)

    return out
