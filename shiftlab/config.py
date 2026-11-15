"""Fixed protocol constants. Nothing in here is tuned on the test partition."""

FS = 360  # Hz, MIT-BIH Arrhythmia Database sampling rate

# AAMI EC57 excludes the four records with paced beats.
PACED = ["102", "104", "107", "217"]

# Inter-patient partition of de Chazal, O'Dwyer & Reilly (IEEE TBME 2004).
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
RECORDS = sorted(DS1 + DS2)

# Records 201 and 202 come from the same subject (MIT-BIH directory), yet the
# standard partition puts 201 in DS1 and 202 in DS2. Subject ids make that explicit.
SUBJECT = {r: r for r in RECORDS}
SUBJECT["202"] = "201"

# AAMI beat classes. Q (paced / unclassifiable) is dropped: after removing the
# paced records it has ~15 beats and cannot be estimated.
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASSES = ["N", "S", "V", "F"]

# Beat window around the annotated R peak: 250 ms before, 450 ms after.
PRE = 90
POST = 162
WIN = PRE + POST

BANDPASS = (0.5, 40.0)  # Hz, zero-phase Butterworth
LOCAL_RR_BEATS = 10
