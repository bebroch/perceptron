field1 = 0.95
field2 = 1.95
field_div = 0.1

POINTS_DATA = [
    (field1, field1, 0),
    (field1, field1+field_div, 0),
    (field1+field_div, field1, 0),
    (field1+field_div, field1+field_div, 0),

    (field1, field2, 0),
    (field1, field2+field_div, 0),
    (field1+field_div, field2, 0),
    (field1+field_div, field2+field_div, 0),

    (field2, field1, 0),
    (field2, field1+field_div, 0),
    (field2+field_div, field1, 0),
    (field2+field_div, field1+field_div, 0),

    (field2, field2, 1),
    (field2, field2+field_div, 1),
    (field2+field_div, field2, 1),
    (field2+field_div, field2+field_div, 1),
]
