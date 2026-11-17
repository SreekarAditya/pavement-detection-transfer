"""Audit the verified 05B review packets locally. Requires Pillow, NumPy,
pandas, pyarrow, Node with WebSocket support, and local Google Chrome.
The private key is read directly from the verified archive, never exported.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
from PIL import Image

OUTPUT = Path(__file__).resolve().parent
TASK = OUTPUT.parent / '05'
DELIVERY = TASK / 'rater_delivery'
ARCHIVE = TASK / 'post_reg_05B_outputs.tar.gz'
PROJECT = OUTPUT.parents[2]
COLLECTIONS = ('ecupdd', 'ird', 'svrdd')
RATERS = ('Aditya', 'Sahasra')
WINDOW = ('crop_left', 'crop_top', 'crop_right', 'crop_bottom')
PAYLOAD = re.compile(rb'(\bconst ITEMS\s*=\s*)(\[.*?\])(;\s*const CATS\s*=)', re.S)
FIELDS = re.compile(r'confidence|(?<![a-z])conf(?![a-z])|predicted[_ -]?class|checkpoint|tertile|'
                    r'agreement|reference[_ -]?(?:box|label)|(?:ground[_ -]?truth|gt)[_ -]?(?:box|label)|'
                    r'e11b_key|crop_(?:left|top|right|bottom)|alert_row|near_iou|unmapped_class|\bpart_a\b|\bstratum\b|\bx[12]\b|\by[12]\b', re.I)
CATEGORIES = {
    'missing mapped distress': 'missing_mapped_distress',
    'other distress': 'other_distress',
    'road object': 'road_object',
    'not damage': 'not_damage',
    'cannot tell': 'cannot_tell',
}
CHROME_TEST = r"""
const [port, file, downloads, rater, categoriesText] = process.argv.slice(1);
const categories = JSON.parse(categoriesText);
const targets = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
const ws = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let id = 0;
const waiting = new Map();
let blockedRequests = 0;
let uncaughtErrors = 0;
ws.onmessage = ({data}) => {
  const msg = JSON.parse(data);
  if (msg.id) {
    const [resolve, reject] = waiting.get(msg.id);
    waiting.delete(msg.id);
    if (msg.error) reject(new Error(msg.error.message)); else resolve(msg.result);
  } else if (msg.method === 'Page.javascriptDialogOpening') {
    command('Page.handleJavaScriptDialog', {accept: true});
  } else if (msg.method === 'Network.requestWillBeSent' && /^https?:/i.test(msg.params.request.url)) {
    blockedRequests++;
  } else if (msg.method === 'Runtime.exceptionThrown') {
    uncaughtErrors++;
  }
};
function command(method, params = {}) {
  return new Promise((resolve, reject) => {
    waiting.set(++id, [resolve, reject]);
    ws.send(JSON.stringify({id, method, params}));
  });
}
async function evalJS(expression) {
  const result = await command('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error('Viewer action raised an exception');
  return result.result.value;
}
await command('Page.enable');
await command('Network.enable');
await command('Network.setBlockedURLs', {urls: ['http://*', 'https://*']});
await command('Runtime.enable');
await command('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloads});
const loaded = await command('Page.navigate', {url: new URL('file://' + file).href});
if (loaded.errorText) throw new Error('Viewer failed to open');
let visible = false;
for (let i = 0; i < 90; i++) {
  await new Promise(resolve => setTimeout(resolve, 1000));
  try {
    visible = await evalJS(`location.protocol === 'file:' &&
      document.querySelector('#cats')?.querySelectorAll('button').length === 5 &&
      [...document.images].every(image => image.complete && image.naturalWidth > 0) &&
      Object.keys(ans).length === 0`);
    if (visible) break;
  } catch {}
}
const result = {offline: Boolean(visible && !blockedRequests && !uncaughtErrors), buttons: false, time: false, export: false};
if (visible) {
  const state = await evalJS(`(() => {
    const map = ${JSON.stringify(categories)};
    const buttons = [...document.querySelectorAll('#cats button')];
    const subclasses = [...document.querySelectorAll('#sub button')];
    let tick = Date.now();
    Date.now = () => tick;
    t0 = tick;
    const expected = {};
    const choices = buttons.filter(button => map[button.textContent] !== 'missing_mapped_distress')
      .map(button => [button, null]);
    const main = buttons.find(button => map[button.textContent] === 'missing_mapped_distress');
    for (const subclass of subclasses) choices.push([main, subclass]);
    let passed = buttons.length === 5 && subclasses.length === 4 && choices.length === 8;
    for (const [index, [button, subclass]] of choices.entries()) {
      const item = ITEMS[i].item, seconds = (index + 1) / 4;
      tick += seconds * 1000;
      button.click();
      if (subclass) {
        passed &&= getComputedStyle(document.querySelector('#sub')).display !== 'none' && !ans[item];
        subclass.click();
      }
      const payload = {category: map[button.textContent], subclass: subclass?.textContent ?? null, seconds};
      expected[item] = payload;
      passed &&= ans[item]?.category === payload.category && ans[item]?.subclass === payload.subclass;
    }
    let timing = Object.keys(expected).length === 8 && Object.entries(expected)
      .every(([item, payload]) => ans[item]?.seconds === payload.seconds && payload.seconds > 0);
    const last = ITEMS[i - 1].item;
    const back = [...document.querySelectorAll('button')].find(button => button.textContent === 'Back');
    back.click();
    timing &&= ITEMS[i].item === last && !!document.querySelector('#st').textContent;
    tick += 500;
    const [button, subclass] = choices.at(-1);
    button.click();
    if (subclass) subclass.click();
    expected[last].seconds += 0.5;
    timing &&= ans[last]?.seconds === expected[last].seconds;
    document.querySelector('button[onclick="save()"]').click();
    return {expected, passed, timing};
  })()`);
  result.buttons = state.passed;
  result.time = state.timing;
  const fs = await import('node:fs/promises');
  const answerFile = downloads + '/answers_' + rater + '.json';
  for (let i = 0; i < 25; i++) {
    try {
      const saved = JSON.parse(await fs.readFile(answerFile, 'utf8'));
      result.export = saved.rater === rater && JSON.stringify(saved.answers) === JSON.stringify(state.expected);
      await fs.unlink(answerFile);
      break;
    } catch { await new Promise(resolve => setTimeout(resolve, 200)); }
  }
}
result.offline &&= !blockedRequests && !uncaughtErrors;
console.log(JSON.stringify(result));
ws.close();
"""


def digest(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive_inputs(path=ARCHIVE, report_path=TASK / 'REPORT.md'):
    report = report_path.read_text()
    match = re.search(r'post_reg_05B_outputs\.tar\.gz` with SHA-256 `([a-f\d]{64})`', report)
    with path.open('rb') as stream:
        if not match or digest(stream) != match[1]:
            raise ValueError('Task 05B archive verification failed')
    allowed = {f'05B/packet_{rater}/index.html' for rater in RATERS}
    seen, copies = set(), {}
    key = None
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name == '05B/private/e11b_key.parquet' and member.isfile():
                key = pd.read_parquet(io.BytesIO(archive.extractfile(member).read()))
            if member.name.startswith(tuple(f'05B/packet_{r}/' for r in RATERS)):
                seen.add(member.name)
                if member.name in allowed and member.isfile():
                    copies[member.name] = digest(archive.extractfile(member))
            if member.name == '05B/RATER_INSTRUCTIONS.md' and member.isfile():
                copies[member.name] = digest(archive.extractfile(member))
    if key is None:
        raise ValueError('Private key is missing from the verified 05B archive')
    return key, copies, seen == allowed


def chrome_binary():
    if configured := os.environ.get('CHROME_BINARY'):
        return configured
    for name in ('google-chrome', 'chromium', 'chromium-browser'):
        if found := shutil.which(name):
            return found
    if shutil.which('mdfind'):
        candidates = subprocess.run(['mdfind', 'kMDItemCFBundleIdentifier == "com.google.Chrome"'],
                                    capture_output=True, text=True, timeout=5, check=False)
        for bundle in candidates.stdout.splitlines():
            binary = Path(bundle) / 'Contents' / 'MacOS' / 'Google Chrome'
            if binary.is_file():
                return str(binary)
    return None


def browser_test(path, rater):
    failure = dict.fromkeys(('offline', 'buttons', 'time', 'export'), False)
    with tempfile.TemporaryDirectory(prefix='packet-browser-test-') as scratch:
        scratch = Path(scratch)
        copy = scratch / 'index.html'
        shutil.copyfile(path, copy)
        downloads = scratch / 'downloads'
        downloads.mkdir()
        profile = scratch / 'chrome'
        chrome = chrome_binary()
        if chrome is None:
            return failure
        command = [chrome, '--headless', '--remote-debugging-port=0', f'--user-data-dir={profile}',
                   '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
                   '--disable-component-update', '--disable-extensions', 'about:blank']
        try:
            with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as browser:
                try:
                    port_file = profile / 'DevToolsActivePort'
                    for _ in range(100):
                        if port_file.is_file():
                            break
                        if browser.poll() is not None:
                            return failure
                        time.sleep(0.1)
                    port = port_file.read_text().splitlines()[0]
                    test = subprocess.run(['node', '--input-type=module', '-e', CHROME_TEST, port,
                                           str(copy), str(downloads), rater, json.dumps(CATEGORIES)],
                                          capture_output=True, text=True, timeout=150, check=True)
                    return json.loads(test.stdout)
                finally:
                    browser.terminate()
        except (OSError, subprocess.SubprocessError, ValueError):
            return failure


def load_packet(path):
    data = path.read_bytes()
    payload = PAYLOAD.search(data)
    if payload is None:
        raise ValueError('Viewer payload is missing')
    code = (data[:payload.start(2)] + b'[]' + data[payload.end(2):]).decode('utf-8')
    return json.loads(payload[2]), code


def value_patterns(key):
    # Sequential IDs are intentionally public. Check private values where
    # they identify an item or model, and exact high-precision numeric tokens.
    private_files = {str(s) for s in key.file}
    distinctive_stems = {Path(s).stem for s in private_files if len(Path(s).stem) >= 12}
    private_numbers = set()
    for field in ('conf', 'near_iou', 'x1', 'y1', 'x2', 'y2', *[c for c in WINDOW if c in key]):
        for value in key[field].dropna():
            for token in (repr(float(value)), f'{value:.8f}'.rstrip('0')):
                if re.fullmatch(r'-?\d+\.\d{7,}', token):
                    private_numbers.add(token)
    private_categories = {str(value) for field in ('collection', 'block', 'part_a', 'unmapped_class')
                          for value in key[field].dropna() if str(value).isidentifier()}
    # Standalone tokens avoid collisions such as "ird" inside "third".
    return private_files | distinctive_stems, private_categories, private_numbers


def contains_private(text, private_values, private_categories, private_numbers):
    text = html.unescape(text)
    code_and_labels = re.sub(r'<style\b[^>]*>.*?</style>', '', text, flags=re.I | re.S)
    identifiers = set(re.findall(r'\b[A-Za-z_]\w*\b', code_and_labels, re.ASCII))
    return (bool(FIELDS.search(text) or re.search(r'\b(?:yolo\w*|rfdetr\w*|seed\d+|[\w.-]+\.pt)\b', text, re.I))
            or any(token in text for token in private_values)
            or bool(private_categories & identifiers)
            or any(token in private_numbers for token in re.findall(r'-?\d+\.\d{7,}', text)))


def open_image(payload):
    data = base64.b64decode(payload, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        # Accept ordinary JFIF encoding fields only. Reject EXIF, XMP,
        # JPEG comments, ICC profiles, and all PNG text/ancillary metadata.
        clean = (image.format in ('JPEG', 'PNG') and not image.getexif()
                 and not getattr(image, 'text', {})
                 and all(marker == 'APP0' and content.startswith(b'JFIF\x00')
                         for marker, content in getattr(image, 'applist', ()))
                 and set(image.info) <= {'jfif', 'jfif_version', 'jfif_unit', 'jfif_density'})
        return image.convert('RGB'), clean


def alert_perimeter(image, box, original, *, magenta):
    pixels = np.asarray(image, dtype=np.int16)
    red, green, blue = np.moveaxis(pixels, 2, 0)
    if magenta:
        colored = (red > 155) & (blue > 145) & (red - green > 55) & (blue - green > 55)
    else:
        colored = (red > 170) & (green > 150) & (red - blue > 55) & (green - blue > 55)
    width, height = image.size
    sx, sy = width / original[0], height / original[1]
    x1, y1, x2, y2 = np.rint(np.asarray(box) * [sx, sy, sx, sy]).astype(int)
    left, right = np.clip([x1, x2], 0, width - 1)
    top, bottom = np.clip([y1, y2], 0, height - 1)
    if right - left < 4 or bottom - top < 4:
        return False
    radius = max(2, round(max(original) / 400 * max(sx, sy)) + 1)
    horizontal = [colored[max(y-radius, 0):y+radius+1, left:right+1].any(axis=0).mean() for y in (top, bottom)]
    vertical = [colored[top:bottom+1, max(x-radius, 0):x+radius+1].any(axis=1).mean() for x in (left, right)]
    # A box touching the image edge can have its outside stroke clipped.
    edge = [(y1 <= 0), (y2 >= height), (x1 <= 0), (x2 >= width)]
    visible = [coverage for coverage, clipped in zip(horizontal + vertical, edge) if not clipped]
    if len(visible) < 2 or min(visible) < 0.55:
        return False

    # Look for a second prominent rectangle of the outline color away from the alert.
    colored[max(top-radius, 0):bottom+radius+1, max(left-radius, 0):right+radius+1] = False
    row_runs = set()
    for y, line in enumerate(colored):
        changes = np.flatnonzero(np.diff(line, prepend=False, append=False))
        for a, b in zip(changes[::2], changes[1::2]):
            if b - a >= 18:
                candidates = [y0 for y0, x0, x1 in row_runs if y-y0 >= 12 and abs(a-x0) <= 2 and abs(b-x1) <= 2]
                for y0 in candidates:
                    interior = colored[y0+4:y-3, a+4:b-4]
                    # Filled paint markings can have rectangular silhouettes.
                    if (interior.mean() < 0.20
                            and colored[y0:y+1, max(a-2, 0):a+3].any(axis=1).mean() > .85
                            and colored[y0:y+1, max(b-3, 0):b+2].any(axis=1).mean() > .85):
                        return False
                row_runs.add((y, a, b))
    return True


def recorded_crop_matches(window, image_size, box, crop_size):
    left, top, right, bottom = window
    x1, y1, x2, y2 = box
    width, height = image_size
    target_w = min(width, max(160, int(np.ceil(2 * (x2 - x1)))))
    target_h = min(height, max(160, int(np.ceil(2 * (y2 - y1)))))
    if (right - left, bottom - top) != (target_w, target_h) or crop_size != (target_w, target_h):
        return False
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return False
    if not (left <= max(0, x1) and min(width, x2) <= right
            and top <= max(0, y1) and min(height, y2) <= bottom):
        return False
    for lo, hi, extent, start, end in ((x1, x2, width, left, right), (y1, y2, height, top, bottom)):
        center_error = (start + end - lo - hi) / 2
        if abs(center_error) > 0.5 and not (
                (center_error > 0 and start == 0) or (center_error < 0 and end == extent)):
            return False
    return True


def image_checks(items, key, properties_dir):
    properties = pd.concat([
        pd.read_parquet(properties_dir / f'{name}.parquet',
                        columns=['file', 'src_w', 'src_h']).assign(collection=name)
        for name in COLLECTIONS
    ]).set_index(['collection', 'file'])
    metadata, neutral, dimensions = True, True, True
    rebuild = set(WINDOW) <= set(key.columns)
    for item in items:
        row = key.loc[item['item']]
        size = properties.loc[(row.collection, row.file)]
        full, full_clean = open_image(item['full'])
        crop, crop_clean = open_image(item['crop'])
        metadata &= full_clean and crop_clean
        box = [row.x1, row.y1, row.x2, row.y2]
        source_size = (int(size.src_w), int(size.src_h))
        neutral &= alert_perimeter(full, box, source_size, magenta=rebuild)
        if rebuild:
            window = tuple(int(row[c]) for c in WINDOW)
            left, top, _, _ = window
            crop_box = (row.x1 - left, row.y1 - top, row.x2 - left, row.y2 - top)
            neutral &= alert_perimeter(crop, crop_box, crop.size, magenta=True)
            dimensions &= recorded_crop_matches(window, source_size, box, crop.size)
        else:
            required = [max(160, 2 * (row.x2 - row.x1)), max(160, 2 * (row.y2 - row.y1))]
            dimensions &= all(actual >= 160 and abs(actual - target) <= 1
                              for actual, target in zip(crop.size, required))
    return metadata, neutral, dimensions


def audit(archive, report, delivery, properties_dir, baseline=None):
    key, copies, clean_members = archive_inputs(archive, report)
    rebuild = set(WINDOW) <= set(key.columns)
    if baseline:
        old_key, _, _ = archive_inputs(baseline, baseline.parent / 'REPORT.md')
        same_sample = rebuild and old_key.equals(key[old_key.columns])
    else:
        same_sample = True
    key = key.set_index('item')
    if not key.index.is_unique:
        raise ValueError('Duplicate items in the private key')
    private_values, private_categories, private_numbers = value_patterns(key)
    instruction = delivery / 'RATER_INSTRUCTIONS.md'
    with instruction.open('rb') as stream:
        expected_copy = digest(stream) == copies.get('05B/RATER_INSTRUCTIONS.md')
    if rebuild:
        expected_copy &= 'magenta outline' in instruction.read_text()
    verified_name = 'Verified archive and byte-identical delivery copies'
    if baseline:
        verified_name += ', original item identities and orders preserved'
    checks = {verified_name: expected_copy}
    orders, pictures = {}, {}
    for rater in RATERS:
        folder = delivery / f'packet_{rater.lower()}'
        path = folder / 'index.html'
        with path.open('rb') as stream:
            checks[verified_name] &= (
                digest(stream) == copies.get(f'05B/packet_{rater}/index.html'))
        checks[f'{rater}: packet contains no key or private files'] = (
            clean_members and not folder.is_symlink() and not path.is_symlink()
            and sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*')) == ['index.html'])
        items, code = load_packet(path)
        order = [item['item'] for item in items]
        verified_ids = len(order) == 450 and len(set(order)) == 450 and set(order) == set(key.index)
        sample = key.reindex(order)
        checks[f'{rater}: 450 items and 150 per four-class collection'] = (
            verified_ids and Counter(sample.collection) == dict.fromkeys(COLLECTIONS, 150))
        checks[f'{rater}: 50 items per confidence tertile per collection'] = (
            verified_ids and Counter(zip(sample.collection, sample.stratum)) ==
            {(c, t): 50 for c in COLLECTIONS for t in range(3)})
        schema = all(set(item) == {'item', 'full', 'crop'} and re.fullmatch(r'I\d{4}', item['item'])
                     for item in items)
        names = ' '.join(p.name for p in folder.rglob('*'))
        checks[f'{rater}: HTML, JavaScript, JSON fields and file names preserve blinding'] = (
            schema and not contains_private(code + ' ' + names + ' ' + instruction.read_text(),
                                            private_values, private_categories, private_numbers))
        orders[rater] = order
        image_ids = {item['item']: tuple(hashlib.sha256(item[part].encode('ascii')).digest()
                                        for part in ('full', 'crop')) for item in items}
        pictures[rater] = image_ids
        if verified_ids and schema:
            if rater == RATERS[0] or image_ids != pictures[RATERS[0]]:
                image_results = image_checks(items, key, properties_dir)
                if rater == RATERS[0]:
                    first_results = image_results
            else:
                image_results = first_results
        else:
            image_results = (False, False, False)
        for name, passed in zip(('Images contain no EXIF or text metadata',
                                 ('One neutral alert outline in both image views' if rebuild
                                  else 'One neutral alert outline on each full image'),
                                 'Crop dimensions are twice the box size and at least 160 pixels'),
                                image_results):
            checks[f'{rater}: {name}'] = passed
        del items
        for label, passed in zip(('Viewer opens offline with images and blank answers',
                                  'Every answer button records',
                                  'Time per item is recorded and accumulates on revisit',
                                  'Save answers exports matching JSON and test file is deleted'),
                                 browser_test(path, rater).values()):
            checks[f'{rater}: {label}'] = passed
    a, b = RATERS
    if baseline:
        checked_raters = set()
        with tarfile.open(baseline, 'r|gz') as old_archive:
            for member in old_archive:
                for rater in RATERS:
                    if member.name == f'05B/packet_{rater}/index.html':
                        payload = PAYLOAD.search(old_archive.extractfile(member).read())
                        prior_order = [i['item'] for i in json.loads(payload[2])]
                        same_sample &= prior_order == orders[rater]
                        checked_raters.add(rater)
        checks[verified_name] &= same_sample and checked_raters == set(RATERS)
    checks['Both raters have identical item images in different orders'] = (
        orders[a] != orders[b] and pictures[a] == pictures[b])
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--report', type=Path, default=TASK / 'REPORT.md')
    parser.add_argument('--delivery', type=Path, default=DELIVERY)
    parser.add_argument('--properties-dir', type=Path, default=PROJECT / '08_lock/root/s2/ext_props')
    parser.add_argument('--baseline', type=Path, help='Verified original archive for item identity and order comparison')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'check_05b.csv')
    args = parser.parse_args()
    checks = audit(args.archive, args.report, args.delivery, args.properties_dir, args.baseline)
    destination = args.output
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        out = csv.writer(stream)
        out.writerow(['check', 'passed yes or no'])
        out.writerows((name, 'yes' if passed else 'no') for name, passed in checks.items())
    os.replace(temporary, destination)
    for name, passed in checks.items():
        if not passed:
            print(f'Failed: {name}')
    print(f'{sum(checks.values())} of {len(checks)} checks passed')
    raise SystemExit(not all(checks.values()))
