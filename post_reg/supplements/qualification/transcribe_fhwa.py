"""Extract the paired table rows and printed answer keys from a layout-text PDF."""
import csv
from pathlib import Path
import re
import subprocess
import sys


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def extract(pdf, target):
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    tables = section(text, "Step 3—Ground Reference Ratings", "Step 4—Vendor Ratings")
    rows = re.findall(r"^\s*([BEGI])\s+(0\.0[35])\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$", tables, re.M)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "fhwa_measurements.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["site", "subsection", "length_miles", "reference_percent", "vendor_p_percent", "vendor_q_percent", "locator"])
        for site in "BEGI":
            group = [r for r in rows if r[0] == site]
            assert len(group) == 10, (site, len(group))
            for i, (_, length, ref, p, q) in enumerate(group, 1):
                writer.writerow([site, i, length, ref, p, q, "Table 5 p. 81" if site in "BE" else "Table 6 p. 82"])
    text = text.split("Step 5—Statistical Equivalence Testing", 1)[1]
    figures = list(re.finditer(r"Figure (21|22|23|24|25|26|30|31)\. Reference Site ([BEGI])[^\n]*Vendor ([PQ])\.", text, re.I))
    assert len(figures) == 8
    with (target / "fhwa_answer_key.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["site","vendor","figure","lower","upper","alpha","decision","difference","difference_sd",
                         "n","vendor_mean","vendor_sd","vendor_se","reference_mean","reference_sd","reference_se",
                         "t_lower","p_lower","t_upper","p_upper","p_governing","note"])
        for f in figures:
            src = text[:f.start()].rsplit("Descriptive Statistics", 1)[-1].replace("−", "-")
            descr = re.findall(r"^\s*(?:Vendor [PQ]|Reference)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", src, re.M)
            diff, sd = re.search(r"Diff\s*=\s*([-\d.]+)\s+StDev\s*=\s*([-\d.]+)", src).groups()
            n, lo_t, lo_p = re.search(r"Difference\s*≤\s*[-\d.]+\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)", src).groups()
            hi_n, hi_t, hi_p = re.search(r"Difference\s*≥\s*[-\d.]+\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)", src).groups()
            p, verdict = re.search(r"greater of the two P-Values is ([-\d.]+)\. (Can(?:not)?) claim equivalence", src).groups()
            alpha = re.search(r"α level:\s*([\d.]+)", src)[1]
            number, site, vendor = f.groups()
            bound = 7.5 if site == "I" else 4
            assert n == hi_n and int(n)+1 == int(descr[0][0]) == int(descr[1][0])
            writer.writerow([site,vendor,number,-bound,bound,alpha,
                             "equivalent" if verdict=="Can" else "equivalence_not_demonstrated",
                             diff,sd,int(n)+1,*descr[0][1:],*descr[1][1:],lo_t,lo_p,hi_t,hi_p,p,
                             "Vendor Q figure labels the descriptive vendor row P" if number=="22" else ""])
    remaining = section(text, "Similarly, the fictional agency conducts", "Step 6—Interpreting")
    with (target/"fhwa_excluded.csv").open("w",newline="") as out:
        writer=csv.writer(out)
        writer.writerow(["site","vendor","published_verdict","locator","reason"])
        for site in "ACDFH":
            p,q=re.search(rf"(?m)^\s*{site}\s+\w+\s+\w+\s+\d+\s+(Not EQ|EQ)\s+\d+\s+(Not EQ|EQ)\s+\d+\s*$",remaining).groups()
            for vendor,value in zip("PQ",(p,q)):
                writer.writerow([site,vendor,value,"Table 7 p. 92","Paired subsection measurements absent from Tables 5 and 6"])


if __name__ == "__main__":
    extract(Path(sys.argv[1]),Path(sys.argv[2]))
