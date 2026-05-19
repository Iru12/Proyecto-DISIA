import argparse
import json
import urllib.error
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valida automaticamente muestras pendientes de revision para una demo."
    )
    parser.add_argument("--base-url", default="http://localhost:18080", help="URL base de la API")
    parser.add_argument("--limit", type=int, default=10, help="Numero de muestras pendientes a validar")
    parser.add_argument("--label", default="generic_scanning", help="Etiqueta validada que se aplicara")
    parser.add_argument("--reviewer", default="demo_auto", help="Nombre del revisor de demo")
    parser.add_argument("--notes", default="Etiqueta simulada para demo", help="Notas de validacion")
    return parser.parse_args()


def main():
    args = parse_args()
    queue_url = f"{args.base_url}/admin/retraining/review-queue?limit={args.limit}"
    label_url = f"{args.base_url}/admin/retraining/labels"

    queue = get_json(queue_url)
    samples = queue.get("pending_samples", [])[: args.limit]

    if not samples:
        print("No hay muestras pendientes para validar.")
        return 0

    validated = 0
    failed = 0

    for sample in samples:
        payload = {
            "sample_id": sample["sample_id"],
            "validated_label": args.label,
            "reviewer": args.reviewer,
            "notes": args.notes,
        }

        try:
            result = post_json(label_url, payload)
            validated += 1
            print(f"OK {result['sample_id']} -> {result['validated_label']}")
        except urllib.error.HTTPError as exc:
            failed += 1
            error_body = exc.read().decode("utf-8", errors="replace")
            print(f"ERROR {sample['sample_id']}: {error_body}")

    print(f"Validadas: {validated}")
    print(f"Errores: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
