# Direction to a sound source from a stereo microphone pair — sign-vote over
# TDOA (band-passed GCC-PHAT, 2-6kHz), NOT ILD (loudness) and NOT a physical angle.
#
# History (see project notes on GCC-PHAT sound localization):
# 1) Original broadband GCC-PHAT (no filter) worked cleanly on an open desk
#    (+-90 deg), but once mounted in the head's ears it produced physically
#    impossible delays (exceeding the c*d limit for the inter-ear baseline) —
#    the open skull skeleton (servos, wiring, neck) leaks sound between the
#    two capsules internally.
# 2) Pivot to ILD (inter-channel level difference) worked at close range
#    (5-6cm), but at real conversational distance (1-3m, in a 3x2.5m room)
#    degraded to near-random sign — room reflections swamp the weak
#    directional loudness difference.
# 3) 2026-08-22: band-pass filtering (2-6kHz, where the internal leak is
#    weaker) before GCC-PHAT did NOT remove the excess delay (still
#    physically impossible), BUT its SIGN correlates reliably with the real
#    direction, and a majority sign-vote over a sliding window of blocks is
#    a usable signal even at 1-3m. Confirmed blind (6/6 correct guesses,
#    hissing and live speech, 0.2-2m range).
#
# angle_deg is NOT a physical angle — it is (right_votes - left_votes) /
# total_votes_in_window * 90 deg, i.e. how confidently the majority of
# blocks in the sliding window lean one way. Accuracy is only "which way
# it's leaning", not degrees. Front/back ambiguity is inherent (as with any
# microphone pair) — a source in front and behind at the same angle are
# indistinguishable; resolve via vision (OAK-D/face_detection).

std_msgs/Header header

float32 angle_deg      # [-90..+90], + = source on the right (by vote), - = left; NOT a physical angle
float32 tdoa_us         # median of the raw (non-physical, inflated by internal leakage) delay over the window, us; sign: + = right, - = left; debug only
float32 confidence      # 0..1, |right_votes - left_votes| / total_votes over the sliding window
float32 rms_dbfs        # loudness of the last block across both channels, dBFS
bool voiced             # last block's RMS passed the energy gate (not silence/noise)
