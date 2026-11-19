from transformers import pipeline


def render_result(res):
    return sorted(
        ((result["sequence"], result["score"]) for result in res),
        key=lambda item: item[1],
        reverse=True,
    )


def print_result(res):
    for sequence, score in res:
        print(f"{sequence} | [{score:.4f}]")


def main():
    prompt = "Samuel est un <mask>!"
    fill_mask = pipeline(
        "fill-mask",
        model="almanach/camembert-base",
        device=-1,  # force CPU
    )

    results = fill_mask(prompt, top_k=5)
    print_result(render_result(results))


if __name__ == "__main__":
    main()
