const diffContent = `Index: C:\\Users\\troyh\\.claude\\plans\\glimmering-meandering-brook.md
===================================================================
--- C:\\Users\\troyh\\.claude\\plans\\glimmering-meandering-brook.md\tOriginal
+++ C:\\Users\\troyh\\.claude\\plans\\glimmering-meandering-brook.md\tWritten
@@ -1,1 +1,1 @@
+File created successfully at: C:\\Users\\troyh\\.claude\\plans\\glimmering-meandering-brook.md`;

const lines = diffContent.split(/\r?\n/);
const result = [];
let inHunk = false;
const hunkHeaderRegex = /^@@ -(\d+),?\d* \+(\d+),?\d* @@/;
let currentOldLine = 0;
let currentNewLine = 0;

for (const line of lines) {
  const hunkMatch = line.match(hunkHeaderRegex);
  if (hunkMatch) {
    currentOldLine = parseInt(hunkMatch[1], 10);
    currentNewLine = parseInt(hunkMatch[2], 10);
    inHunk = true;
    result.push({ type: 'hunk', content: line });
    currentOldLine--;
    currentNewLine--;
    continue;
  }
  if (!inHunk) {
    continue;
  }
  if (line.startsWith('+')) {
    currentNewLine++;
    result.push({ type: 'add', newLine: currentNewLine, content: line.substring(1) });
  } else if (line.startsWith('-')) {
    currentOldLine++;
    result.push({ type: 'del', oldLine: currentOldLine, content: line.substring(1) });
  } else if (line.startsWith(' ')) {
    currentOldLine++;
    currentNewLine++;
    result.push({ type: 'context', oldLine: currentOldLine, newLine: currentNewLine, content: line.substring(1) });
  }
}

console.log('Parsed lines:', result.length);
const displayableLines = result.filter(l => l.type !== 'hunk' && l.type !== 'other');
console.log('Displayable lines:', displayableLines.length);
