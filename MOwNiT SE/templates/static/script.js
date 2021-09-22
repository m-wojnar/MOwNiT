const methods = ["vsr-idf", "vsr", "edlsi", "lsi"];

document.getElementById("search-input").addEventListener("keyup", (event) => {
  if (event.code === 13) 
    search();
}); 

function setMethod(name) {
  showDot();
  document.cookie = `${name}; path=/`;
  showDot();
}

function getMethod() {
  for (let i = 0; i < methods.length; i += 1) {
    if (document.cookie.search(methods[i]) !== -1) 
      return methods[i];
  }
  
  return "edlsi";
}

function search() {
  let text = document.getElementById("search-input").value;
  let method = getMethod();

  if (text.trim().length == 0) 
    return;
  
  window.open(encodeURI(`/search/${method}/${text}`), '_self');
}

function lucky() {
  let text = document.getElementById("search-input").value;
  let method = getMethod();

  if (text.trim().length == 0) 
    return;
  
  window.open(encodeURI(`/lucky/${method}/${text}`), '_self');
}

function showDot() {
  for (let t of methods) 
    document.getElementById(t + "-dot").className = "d-none";

  document.getElementById(getMethod() + "-dot").classList.remove("d-none");
}

showDot();
