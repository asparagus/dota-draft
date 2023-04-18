import datetime
from draft.data.filter import DateFilter


PATCH_732e = DateFilter(
    start_date=datetime.date(2023, 3, 6),
    end_date=None,
)

PATCH_732d = DateFilter(
    start_date=datetime.date(2022, 11, 29),
    end_date=datetime.date(2023, 3, 6),
)

PATCH_732c = DateFilter(
    start_date=datetime.date(2022, 9, 27),
    end_date=datetime.date(2022, 11, 29),
)
