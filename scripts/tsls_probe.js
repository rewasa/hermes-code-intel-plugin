// TSLS initialize-handshake probe: prints exactly what the server answers
// (or "NO RESPONSE"), bounded, non-interactive. Used by CI to diagnose why
// typescript-language-server --stdio never answers initialize on runners.
const { spawn } = require("child_process");
const p = spawn("typescript-language-server", ["--stdio"], { cwd: process.cwd() });
let got = false;
p.stdout.on("data", d => { got = true; console.log("TSLS-OUT:", d.toString().slice(0, 400)); });
p.stderr.on("data", d => console.log("TSLS-ERR:", d.toString().slice(0, 400)));
p.on("exit", (c, s) => console.log("TSLS-EXIT:", c, s));
const init = JSON.stringify({jsonrpc:"2.0",id:1,method:"initialize",params:{processId:null,rootUri:"file://" + process.cwd(),capabilities:{}}});
const msg = "Content-Length: " + Buffer.byteLength(init) + "\r\n\r\n" + init;
p.stdin.write(msg);
setTimeout(() => {
  if (!got) { console.log("TSLS: NO RESPONSE in 8s"); p.kill("SIGKILL"); process.exit(1); }
  console.log("TSLS: replied OK");
  p.kill("SIGKILL");
  process.exit(0);
}, 8000);
