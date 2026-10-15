"""Defer heavyweight OCR until the scheduler actually needs text evidence."""
from .runtime_service_cache import get_or_create_service


class LazyOCR:
    def __init__(self, params):
        self.params = dict(params)

    def _service(self):
        from .ocr_service import PaddleOCRService
        return get_or_create_service("paddle_ocr", self.params, lambda: PaddleOCRService(**self.params))

    def read_image(self, *args, **kwargs):
        return self._service().read_image(*args, **kwargs)

    def read_region(self, *args, **kwargs):
        return self._service().read_region(*args, **kwargs)


def build_lazy_ocr(params):
    return get_or_create_service("lazy_ocr", params, lambda: LazyOCR(params))
