from src.models import StudentSearchResults


def find_question_id_index(qid: str,
                           search_results: StudentSearchResults) -> int:
    for i, search in enumerate(search_results.search_results):
        if search.question_id == qid:
            return i
    return -1


def calculate_IoU(start_a: int, end_a: int,
                  start_b: int, end_b: int) -> float:
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = intersection_end - intersection_start

    if intersection <= 0:
        return 0.0

    size_a = end_a - start_a
    size_b = end_b - start_b
    union = size_a + size_b - intersection

    iou = intersection / union
    return iou
