"""Eurostat's new passenger-car registrations, all fuels and engines, in app codes."""

from __future__ import annotations

import pytest

from icor.evidence.eurostat_new_registrations import NewRegistrations, includes_used_imports, parse


def _document(values, status=None):
    dims = {
        "freq": ["A"],
        "unit": ["NR"],
        "mot_nrg": ["TOTAL", "DIE"],
        "engine": ["TOTAL"],
        "geo": ["EL", "PL"],
        "time": ["2012", "2013"],
    }
    order = list(dims)
    sizes = [len(dims[name]) for name in order]

    def position(**coordinates):
        flat = 0
        for name, size in zip(order, sizes, strict=True):
            flat = flat * size + dims[name].index(coordinates.get(name, dims[name][0]))
        return str(flat)

    return {
        "id": order,
        "size": sizes,
        "dimension": {
            name: {"category": {"index": {item: index for index, item in enumerate(items)}}}
            for name, items in dims.items()
        },
        "value": {position(**key): value for key, value in values},
        "status": {position(**key): flag for key, flag in (status or [])},
    }


def test_totals_are_read_per_market_and_year_with_their_flags():
    values = [
        ({"geo": "EL", "time": "2012"}, 60),
        ({"geo": "EL", "time": "2013"}, 70),
        ({"geo": "EL", "time": "2013", "mot_nrg": "DIE"}, 30),
        ({"geo": "PL", "time": "2013"}, 300),
    ]
    status = [({"geo": "EL", "time": "2012"}, "d")]

    result = parse(_document(values, status), {"GR", "PL"})

    assert result == {
        "GR": {2012: NewRegistrations(60, "d"), 2013: NewRegistrations(70)},
        "PL": {2013: NewRegistrations(300)},
    }


def test_a_document_without_totals_is_refused():
    document = _document([])
    document["dimension"]["mot_nrg"] = {"category": {"index": {"DIE": 0}}}

    with pytest.raises(ValueError, match="mot_nrg TOTAL"):
        parse(document, {"PL"})


def test_used_imports_are_flagged_before_2013_where_offices_counted_them():
    assert includes_used_imports("PL", 2012) is True
    assert includes_used_imports("PL", 2013) is False
    assert includes_used_imports("DE", 2010) is False
