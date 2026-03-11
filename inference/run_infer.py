from inference.model_loader import load_model
from inference.svg_processor import process_svg


def main():

    model, scaler = load_model()

    process_svg(
        model,
        scaler,
        "input.svg",
        "text_paths.dxf"
    )


if __name__ == "__main__":
    main()