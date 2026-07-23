import crypto from "node:crypto";

function levenshtein(a, b, equals = (left, right) => left === right) {
  if (a.length < b.length) return levenshtein(b, a, (left, right) => equals(right, left));
  let previous = new Uint32Array(b.length + 1);
  let current = new Uint32Array(b.length + 1);
  for (let column = 0; column <= b.length; column += 1) previous[column] = column;
  for (let row = 1; row <= a.length; row += 1) {
    current[0] = row;
    for (let column = 1; column <= b.length; column += 1) {
      current[column] = Math.min(
        previous[column] + 1,
        current[column - 1] + 1,
        previous[column - 1] + (equals(a[row - 1], b[column - 1]) ? 0 : 1),
      );
    }
    [previous, current] = [current, previous];
  }
  return previous[b.length];
}

export function normalizeText(text) {
  return String(text ?? "").normalize("NFKC");
}

export function words(text) {
  return normalizeText(text)
    .toLocaleLowerCase("und")
    .match(/[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}\p{N}]+)*/gu) ?? [];
}

export function characters(text) {
  return [...normalizeText(text)];
}

export function errorRate(reference, candidate, tokenizer) {
  const left = tokenizer(reference);
  const right = tokenizer(candidate);
  const edits = levenshtein(left, right);
  return {
    edits,
    referenceUnits: left.length,
    candidateUnits: right.length,
    rate: left.length === 0 ? (right.length === 0 ? 0 : 1) : edits / left.length,
  };
}

export function transcriptMetrics(reference, candidate) {
  return {
    wer: errorRate(reference, candidate, words),
    cer: errorRate(reference, candidate, characters),
  };
}

export function sha256Json(value) {
  return crypto.createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

export function tokenRecords(run) {
  const tokenEvents = (run.events ?? []).filter((event) => event.type === "token");
  return (run.tokens ?? []).map((id, index) => {
    const event = tokenEvents[index] ?? {};
    return {
      index,
      id,
      piece: event.piece ?? "",
      decoderPosition: event.decoderPosition ?? null,
      audioEndSample: event.audioEndSample ?? null,
      timestampMs: Number.isFinite(event.audioEndSample) ? event.audioEndSample / 16 : null,
      special: event.special ?? false,
    };
  });
}

/**
 * Exact Levenshtein alignment with one byte of backtrace per matrix cell.
 * Two token rows plus the backtrace are bounded by O(reference*candidate);
 * local fixtures are about 1.5k/3k tokens, so this remains modest and exact.
 */
export function alignTokenRecords(reference, candidate) {
  const rows = reference.length + 1;
  const columns = candidate.length + 1;
  const directions = new Uint8Array(rows * columns);
  let previous = new Uint32Array(columns);
  let current = new Uint32Array(columns);
  for (let column = 1; column < columns; column += 1) {
    previous[column] = column;
    directions[column] = 2;
  }
  for (let row = 1; row < rows; row += 1) {
    current[0] = row;
    directions[row * columns] = 1;
    for (let column = 1; column < columns; column += 1) {
      const substitution = previous[column - 1] +
        (reference[row - 1].id === candidate[column - 1].id ? 0 : 1);
      const deletion = previous[column] + 1;
      const insertion = current[column - 1] + 1;
      if (substitution <= deletion && substitution <= insertion) {
        current[column] = substitution;
        directions[row * columns + column] = 0;
      } else if (deletion <= insertion) {
        current[column] = deletion;
        directions[row * columns + column] = 1;
      } else {
        current[column] = insertion;
        directions[row * columns + column] = 2;
      }
    }
    [previous, current] = [current, previous];
  }

  const reversed = [];
  let row = reference.length;
  let column = candidate.length;
  while (row > 0 || column > 0) {
    const direction = directions[row * columns + column];
    if (row > 0 && column > 0 && direction === 0) {
      const left = reference[row - 1];
      const right = candidate[column - 1];
      reversed.push({
        type: left.id === right.id ? "equal" : "substitution",
        reference: left,
        candidate: right,
      });
      row -= 1;
      column -= 1;
    } else if (row > 0 && (column === 0 || direction === 1)) {
      reversed.push({ type: "deletion", reference: reference[row - 1], candidate: null });
      row -= 1;
    } else {
      reversed.push({ type: "insertion", reference: null, candidate: candidate[column - 1] });
      column -= 1;
    }
  }
  const operations = reversed.reverse();
  return {
    distance: previous[candidate.length],
    operations,
  };
}

function removePunctuation(text) {
  return normalizeText(text).replace(/[\p{P}\p{S}\s]+/gu, "");
}

const NEGATIONS = new Set([
  "no", "not", "never", "none", "neither", "nor", "nothing", "nobody",
  "нет", "не", "ни", "никогда", "ничего", "никто",
  "non", "ne", "pas", "jamais", "kein", "keine", "nicht", "nie",
  "no", "nunca", "jamás",
]);

function negations(text) {
  return words(text).filter((word) => NEGATIONS.has(word) || word.endsWith("n't"));
}

function numbers(text) {
  return normalizeText(text).match(/\p{N}+(?:[.,:/-]\p{N}+)*/gu) ?? [];
}

function commonPrefix(left, right) {
  let index = 0;
  while (index < left.length && index < right.length && left[index] === right[index]) index += 1;
  return index;
}

export function classifyDivergence(referenceText, candidateText, operationTypes = []) {
  const left = normalizeText(referenceText);
  const right = normalizeText(candidateText);
  if (operationTypes.every((type) => type === "insertion")) return "word insertion";
  if (operationTypes.every((type) => type === "deletion")) return "word deletion";
  if (left.trim() === right.trim()) return "whitespace";
  if (removePunctuation(left) === removePunctuation(right)) return "punctuation";
  if (JSON.stringify(numbers(left)) !== JSON.stringify(numbers(right))) return "number";
  if (JSON.stringify(negations(left)) !== JSON.stringify(negations(right))) return "negation";

  const leftWords = words(left);
  const rightWords = words(right);
  if (leftWords.length === 1 && rightWords.length === 1) {
    if (/^\p{Lu}/u.test(left.trim()) || /^\p{Lu}/u.test(right.trim())) return "proper noun";
    const prefix = commonPrefix(leftWords[0], rightWords[0]);
    if (prefix >= Math.min(leftWords[0].length, rightWords[0].length) * 0.6) return "inflection";
    const spellingDistance = levenshtein([...leftWords[0]], [...rightWords[0]]);
    if (spellingDistance <= Math.max(2, Math.ceil(leftWords[0].length * 0.25))) return "spelling";
  }
  return "word substitution";
}

function localText(records) {
  return records.map((record) => record?.piece ?? "").join("");
}

function trailingWord(text) {
  const withoutTrailingPunctuation = normalizeText(text)
    .replace(/[^\p{L}\p{M}\p{N}]+$/gu, "");
  return withoutTrailingPunctuation.match(/[\p{L}\p{M}\p{N}]+$/u)?.[0] ?? "";
}

export function divergenceRegions(reference, candidate) {
  const alignment = alignTokenRecords(reference, candidate);
  const regions = [];
  let current = [];
  let equalRunAfter = 0;

  const flush = () => {
    if (current.length === 0) return;
    const referenceRecords = current.flatMap((operation) =>
      operation.reference ? [operation.reference] : []);
    const candidateRecords = current.flatMap((operation) =>
      operation.candidate ? [operation.candidate] : []);
    const firstReference = referenceRecords[0]?.index ??
      Math.max(0, (candidateRecords[0]?.index ?? 0) - 1);
    const lastReference = referenceRecords.at(-1)?.index ?? firstReference;
    const firstCandidate = candidateRecords[0]?.index ??
      Math.max(0, (referenceRecords[0]?.index ?? 0) - 1);
    const lastCandidate = candidateRecords.at(-1)?.index ?? firstCandidate;
    const referenceText = localText(referenceRecords);
    const candidateText = localText(candidateRecords);
    const timestampRecord = candidateRecords[0] ?? referenceRecords[0] ?? null;
    const contextBeforeReference =
      reference.slice(Math.max(0, firstReference - 5), firstReference);
    const contextAfterReference =
      reference.slice(lastReference + 1, lastReference + 11);
    const contextBeforeCandidate =
      candidate.slice(Math.max(0, firstCandidate - 5), firstCandidate);
    const contextAfterCandidate =
      candidate.slice(lastCandidate + 1, lastCandidate + 11);
    let classification = classifyDivergence(
      referenceText,
      candidateText,
      current.map((operation) => operation.type),
    );
    if (["inflection", "spelling", "word substitution"].includes(classification)) {
      const referenceWord = trailingWord(localText([
        ...contextBeforeReference,
        ...referenceRecords,
      ]));
      const candidateWord = trailingWord(localText([
        ...contextBeforeCandidate,
        ...candidateRecords,
      ]));
      if (/^\p{Lu}/u.test(referenceWord) || /^\p{Lu}/u.test(candidateWord)) {
        classification = "proper noun";
      }
    }
    const meaningChanged = ["punctuation", "whitespace"].includes(classification)
      ? false
      : ["number", "negation"].includes(classification)
        ? true
        : null;
    regions.push({
      referenceStart: firstReference,
      referenceEnd: lastReference,
      candidateStart: firstCandidate,
      candidateEnd: lastCandidate,
      referenceIds: referenceRecords.map((record) => record.id),
      candidateIds: candidateRecords.map((record) => record.id),
      referencePieces: referenceRecords.map((record) => record.piece),
      candidatePieces: candidateRecords.map((record) => record.piece),
      contextBeforeReference,
      contextAfterReference,
      contextBeforeCandidate,
      contextAfterCandidate,
      audioPosition: timestampRecord?.audioEndSample ?? null,
      timestampMs: timestampRecord?.timestampMs ?? null,
      localReferenceText: localText(reference.slice(Math.max(0, firstReference - 5), lastReference + 11)),
      localCandidateText: localText(candidate.slice(Math.max(0, firstCandidate - 5), lastCandidate + 11)),
      classification,
      meaningChanged,
      reconverged: false,
      operations: current.map((operation) => operation.type),
    });
    current = [];
    equalRunAfter = 0;
  };

  for (const operation of alignment.operations) {
    if (operation.type === "equal") {
      if (current.length > 0) {
        equalRunAfter += 1;
        if (equalRunAfter >= 1) {
          flush();
          regions.at(-1).reconverged = true;
        }
      }
    } else {
      if (equalRunAfter > 0) flush();
      current.push(operation);
    }
  }
  flush();

  return {
    tokenDistance: alignment.distance,
    tokenReferenceCount: reference.length,
    tokenCandidateCount: candidate.length,
    tokenDivergenceRate: reference.length === 0
      ? (candidate.length === 0 ? 0 : 1)
      : alignment.distance / reference.length,
    firstDivergence: regions[0]?.referenceStart ?? null,
    lastDivergence: regions.at(-1)?.referenceEnd ?? null,
    // A final local edit cannot demonstrate reconvergence because no later
    // token exists, but it is not thereby "sustained". Classify sustained
    // desynchronization by the aligned divergent span, not by end-of-file.
    sustainedDesynchronization: regions.some((region) =>
      Math.max(region.referenceEnd - region.referenceStart + 1,
        region.candidateEnd - region.candidateStart + 1) > 10),
    regions,
  };
}

export function sentenceList(text) {
  const normalized = normalizeText(text).trim();
  if (!normalized) return [];
  if (typeof Intl.Segmenter === "function") {
    const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });
    return [...segmenter.segment(normalized)]
      .map((entry) => entry.segment.trim())
      .filter(Boolean);
  }
  return normalized.split(/(?<=[.!?。！？])\s+/u).filter(Boolean);
}

export function semanticRisk(referenceText, candidateText, divergence) {
  const referenceNumbers = numbers(referenceText);
  const candidateNumbers = numbers(candidateText);
  const referenceNegations = negations(referenceText);
  const candidateNegations = negations(candidateText);
  const referenceSentences = sentenceList(referenceText);
  const candidateSentences = sentenceList(candidateText);
  return {
    changedNumbers: JSON.stringify(referenceNumbers) !== JSON.stringify(candidateNumbers),
    referenceNumbers,
    candidateNumbers,
    changedNegations: JSON.stringify(referenceNegations) !== JSON.stringify(candidateNegations),
    referenceNegations,
    candidateNegations,
    sentenceCountChanged: referenceSentences.length !== candidateSentences.length,
    referenceSentenceCount: referenceSentences.length,
    candidateSentenceCount: candidateSentences.length,
    sustainedTokenDesynchronization: divergence.sustainedDesynchronization,
    lexicalRegions: divergence.regions.filter((region) =>
      !["punctuation", "whitespace"].includes(region.classification)).length,
  };
}

export function transcriptWordDiff(referenceText, candidateText) {
  const left = normalizeText(referenceText).split(/(\s+)/u).filter((part) => part.length > 0);
  const right = normalizeText(candidateText).split(/(\s+)/u).filter((part) => part.length > 0);
  const reference = left.map((piece, index) => ({ id: piece, piece, index }));
  const candidate = right.map((piece, index) => ({ id: piece, piece, index }));
  const alignment = alignTokenRecords(reference, candidate).operations;
  const lines = ["--- F32/F32 oracle", "+++ candidate"];
  for (const operation of alignment) {
    if (operation.type === "equal") lines.push(` ${operation.reference.piece}`);
    else if (operation.type === "deletion") lines.push(`-${operation.reference.piece}`);
    else if (operation.type === "insertion") lines.push(`+${operation.candidate.piece}`);
    else {
      lines.push(`-${operation.reference.piece}`);
      lines.push(`+${operation.candidate.piece}`);
    }
  }
  return lines.join("\n");
}
