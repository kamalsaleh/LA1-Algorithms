function parseMatrixEntriesFromString(M) {
  M = M.replace(/\n/g, "").replace(/\r/g, "");
  M = M.replace(/\s+/g, " ").trim();
  if (M.endsWith(";")) {
    M = M.slice(0, -1);
  }
  
  if (M.includes("[") && M.includes("]")) {
    M = M.replace(/\s+/g, "")
    .replace(/\],\[/g, ";")
    .replace(/\[\[/g, "")
    .replace(/\]\]/g, "")
    .replace(/,/g, " ");
  }
  
  M = M.split(";").map(row =>
    row.trim().split(" ")
  );
  
  return M[0].map((_, colIndex) => M.map(row => row[colIndex]))
}

function transposeMContent(name) {
  var M = document.querySelector('[name="' + name + '"]').value;
  
  M = parseMatrixEntriesFromString(M);
  
  M = "[" + M.map(row => "[ " + row.join(", ") + " ]").join(",\n ") + "]";
  
  document.querySelector('[name="' + name + '"]').value = M;
}

function toggleChar() {
  var ring = document.getElementById("specify-ring");
  var field_of_polynomial_ring = document.getElementById("field-of-polynomial-ring");
  var domain = document.getElementById("specify-domain");
  var char = document.getElementById("specify-field-characteristic");
  
  
  if (ring.value === "Integers" || ring.value === "Rationals") {
    field_of_polynomial_ring.style.display = "none";
    char.style.display = "none";
  }
  
  
  if (ring.value === "FiniteField") {
    field_of_polynomial_ring.style.display = "none";
    char.style.display = "block";
  }
  
  if (ring.value === "PolynomialRing") {
    field_of_polynomial_ring.style.display = "block";
    
    if (domain.value === "FiniteField") {
      char.style.display = "block";
    } else {
      char.style.display = "none";
    }
  }
}

function toggleComputation() {
  const details = document.getElementById("computation-details");
  const button = document.getElementById("toggle-button");
  
  if (details.style.display === "none") {
    details.style.display = "block";
    button.textContent = "Hide Computation";
  } else {
    details.style.display = "none";
    button.textContent = "Show Computation";
  }
}

function generateIdentityMatrix(rank) {
  var M = "";
  
  for (var i = 0; i < rank; i++) {
    if (i > 0) {
      M += " [ ";
    } else {
      M += "[ ";
    }
    for (var j = 0; j < rank; j++) {
      if (i === j) {
        M += "1";
      } else {
        M += "0";
      }
      
      if (j < rank - 1) {
        M += ", ";
      }
    }
    M += " ]";
    
    if (i < rank - 1) {
      M += ",\n";
    }
  }
  
  if (rank > 0) {
    M = "[" + M + "]";
  }
  
  return M;
}

function identityMatrix(name) {
  var rank = document.getElementById("identity-matrix-rank-" + name).value;

  M = generateIdentityMatrix(rank);
  document.querySelector('[name="' + name + '"]').value = M;
}

function toggleReduceRows() {
  var art_of_reduction = document.getElementById("art-of-reduction");
  var reduce_the_first_n_rows = document.getElementById("reduce-the-first-n-rows");
  
  if (art_of_reduction.value === "REF") {
    reduce_the_first_n_rows.style.display = "none";
  } else {
    reduce_the_first_n_rows.style.display = "block";
  }
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(function() {
    alert('Output copied to clipboard');
  }, function(err) {
    console.error('Could not copy text: ', err);
    alert('Failed to copy output to clipboard');
  });
}

function nextPrime() {
  var n = parseInt(document.getElementById("char-input").value);
  var p = n;
  
  while (true) {
    var isPrime = true;
    
    for (var i = 2; i <= Math.sqrt(p); i++) {
      if (p % i === 0) {
        isPrime = false;
        break;
      }
    }
    
    if (isPrime) {
      break;
    }
    
    p++;
  }
  
  document.getElementById("char-input").value = p;
}