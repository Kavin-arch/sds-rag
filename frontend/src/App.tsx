import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Activity, ArrowDown, ArrowUp, Check, ChevronDown, CircleHelp, Copy, FileText, FlaskConical, FolderOpen, Layers2, LoaderCircle, Menu, MessageSquareText, Plus, Search, ShieldCheck, Sparkles, X, Zap } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { api, type ChatResponse, type SdsDocument, type Source } from './lib/api'

type Message = { id: string; role: 'user' | 'assistant'; content: string; sources?: Source[]; retrieval?: ChatResponse['retrieval']; pending?: boolean; error?: boolean }
type SearchMode = 'hybrid' | 'dense' | 'sparse'
const suggestions = [
  { icon: ShieldCheck, label: 'Emergency response', question: 'What first aid measures should I take after exposure?' },
  { icon: Activity, label: 'Safe handling', question: 'What are the safe handling and storage requirements?' },
  { icon: FlaskConical, label: 'Chemical properties', question: 'What are the physical and chemical properties?' },
]
const formatName = (filename: string) => filename.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
const shortKey = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl'

function App() {
  const [documents, setDocuments] = useState<SdsDocument[]>([])
  const [totalSections, setTotalSections] = useState(0)
  const [totalChunks, setTotalChunks] = useState(0)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [toast, setToast] = useState('')
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [mobileNav, setMobileNav] = useState(false)
  const [activeDocument, setActiveDocument] = useState('')
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [mode, setMode] = useState<SearchMode>('hybrid')
  const [drawerSource, setDrawerSource] = useState<Source | null>(null)
  const [highlightedSource, setHighlightedSource] = useState('')
  const [traceOpen, setTraceOpen] = useState<string | null>(null)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const inventoryInitialized = useRef(false)

  const refresh = useCallback(async () => {
    try {
      const [health, inventory] = await Promise.all([api.health(), api.documents()])
      setConnected(health.status === 'running')
      setTotalChunks(health.chunks_stored)
      setDocuments(inventory.documents)
      setTotalSections(inventory.total_sections)
      setSelectedDocs((current) => inventoryInitialized.current ? current.filter((name) => inventory.documents.some((doc) => doc.filename === name)) : inventory.documents.map((doc) => doc.filename))
      inventoryInitialized.current = true
    } catch { setConnected(false) }
  }, [])

  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 30_000); return () => window.clearInterval(timer) }, [refresh])
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight }, [messages, busy])
  useEffect(() => { if (textareaRef.current) { textareaRef.current.style.height = 'auto'; textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 170)}px` } }, [question])
  useEffect(() => {
    const onShortcut = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); setPaletteOpen((open) => !open); return }
      if (event.key === 'Escape') { setPaletteOpen(false); setDrawerSource(null) }
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === 'c') { event.preventDefault(); void copyLatest() }
    }
    window.addEventListener('keydown', onShortcut)
    return () => window.removeEventListener('keydown', onShortcut)
  })

  async function copyLatest() {
    const last = [...messages].reverse().find((message) => message.role === 'assistant' && !message.pending && !message.error)
    if (!last) return
    try { await navigator.clipboard.writeText(last.content); setCopied(true); setToast('Answer copied to clipboard'); window.setTimeout(() => setCopied(false), 1500); window.setTimeout(() => setToast(''), 2600) } catch { setToast('Clipboard access unavailable'); window.setTimeout(() => setToast(''), 2600) }
  }

  async function ask(raw = question, deep = false) {
    const value = raw.trim()
    if (!value || busy) return
    if (documents.length && selectedDocs.length === 0) { setError('Select at least one knowledge source to search.'); return }
    setError(''); setQuestion(''); setBusy(true)
    const userMessage: Message = { id: crypto.randomUUID(), role: 'user', content: value }
    const pending: Message = { id: crypto.randomUUID(), role: 'assistant', content: '', pending: true }
    setMessages((current) => [...current, userMessage, pending])
    try {
      const selected = activeDocument && selectedDocs.includes(activeDocument) ? [activeDocument] : selectedDocs
      const result: ChatResponse = await api.chat(value, selected, mode, deep)
      setMessages((current) => current.map((item) => item.id === pending.id ? { ...item, content: result.answer, sources: result.sources, retrieval: result.retrieval, pending: false } : item))
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : 'Something went wrong.'
      setMessages((current) => current.map((item) => item.id === pending.id ? { ...item, content: message, pending: false, error: true } : item))
    } finally { setBusy(false); textareaRef.current?.focus() }
  }
  async function onSubmit(event: FormEvent) { event.preventDefault(); await ask() }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); void ask(question, true); return }
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void ask() }
  }
  async function onFile(file?: File) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) { setError('Choose a PDF file to add it to your library.'); return }
    setUploading(true); setError(''); setToast('')
    try { const result = await api.upload(file); setToast(`${formatName(result.filename)} indexed successfully`); await refresh(); window.setTimeout(() => setToast(''), 4000) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not upload this PDF.') }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = '' }
  }
  const filteredDocuments = documents.filter((document) => formatName(document.filename).toLowerCase().includes(search.toLowerCase()))
  const toggleDocument = (filename: string) => setSelectedDocs((current) => current.includes(filename) ? current.filter((name) => name !== filename) : [...current, filename])
  const modeName = mode === 'hybrid' ? 'Hybrid' : mode === 'dense' ? 'Dense' : 'Sparse'
  const renderedAnswer = (message: Message) => message.content.replace(/\[(S\d+)\]/g, '[$1](citation:$1)')
  const highlightText = (text: string, query: string) => {
    const terms = [...new Set(query.toLowerCase().match(/[a-z0-9-]{3,}/g) ?? [])]
    if (!terms.length) return text
    const pattern = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
    return text.split(pattern).map((part, index) => terms.includes(part.toLowerCase()) ? <mark key={index}>{part}</mark> : part)
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
      <div className="brand-row"><div className="brand-mark"><Layers2 size={17} /></div><div className="brand-name">atlas<span>.</span><small>KNOWLEDGE SYSTEMS</small></div><button className="icon-button mobile-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={16} /></button></div>
      <button className="workspace-switch"><div className="workspace-icon">N</div><div className="workspace-copy"><strong>Northstar Labs</strong><span>Enterprise workspace</span></div><ChevronDown size={14} className="muted-icon" /></button>
      <button className="new-chat" onClick={() => { setMessages([]); setQuestion(''); setError(''); setMobileNav(false) }}><Plus size={15} /><span>New conversation</span><kbd>{shortKey} K</kbd></button>
      <div className="nav-caption">WORKSPACE</div>
      <button className="nav-item nav-item-active"><MessageSquareText size={15} /><span>Ask knowledge base</span></button>
      <button className="nav-item" onClick={() => fileInput.current?.click()}><FolderOpen size={15} /><span>Source library</span><span className="nav-count">{documents.length}</span></button>
      <div className="library-heading"><div className="nav-caption">KNOWLEDGE SOURCES <span>{documents.length}</span></div><button className="icon-button tiny" onClick={() => fileInput.current?.click()} aria-label="Add a source"><Plus size={15} /></button></div>
      <label className="search-box"><Search size={13} /><input placeholder="Filter sources" value={search} onChange={(event) => setSearch(event.target.value)} />{search && <button onClick={() => setSearch('')} aria-label="Clear search"><X size={12} /></button>}</label>
      <div className="source-caption">PDF DOCUMENTS <span className="source-sync"><i /> LIVE</span></div>
      <div className="document-list">
        {filteredDocuments.map((document) => <div className={`document-item ${activeDocument === document.filename ? 'document-active' : ''}`} key={document.filename}>
          <button className="source-toggle" onClick={() => toggleDocument(document.filename)} aria-label={`${selectedDocs.includes(document.filename) ? 'Exclude' : 'Include'} ${formatName(document.filename)} from search`}><span className={`checkbox ${selectedDocs.includes(document.filename) ? 'checked' : ''}`}>{selectedDocs.includes(document.filename) && <Check size={10} />}</span></button>
          <button className="document-main" onClick={() => { setActiveDocument(activeDocument === document.filename ? '' : document.filename); setMobileNav(false) }}><span className="pdf-icon"><FileText size={13} /></span><span className="document-copy"><strong>{formatName(document.filename)}</strong><small>{document.sections} sections · Indexed</small></span></button>
        </div>)}
        {filteredDocuments.length === 0 && <div className="empty-documents">{search ? 'No matching sources' : 'Your knowledge base is empty'}</div>}
      </div>
      <button className="add-document" onClick={() => fileInput.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle className="spin" size={14} /> : <Plus size={14} />}<span>{uploading ? 'Indexing source…' : 'Add knowledge source'}</span></button>
      <div className="sidebar-bottom"><div className="indexed-card"><div className="indexed-head"><span className="indexed-dot" /><span>INDEX STATUS</span><span className={`indexed-live ${connected ? '' : 'offline'}`}>{connected ? 'HEALTHY' : 'OFFLINE'}</span></div><div className="indexed-number">{totalChunks.toLocaleString()}<span> chunks indexed</span></div><div className="indexed-bar"><i style={{ width: `${Math.min(totalChunks / 100, 100)}%` }} /></div><div className="indexed-foot"><span>{totalSections.toLocaleString()} sections</span><span>{documents.length} sources</span></div></div><div className="profile-row"><div className="avatar">KP</div><div className="profile-copy"><strong>Kavin P</strong><span>Workspace admin</span></div><button className="icon-button tiny" aria-label="Profile menu"><ChevronDown size={14} /></button></div></div>
    </aside>
    {mobileNav && <button className="mobile-scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}
    <main className="main-area">
      <header className="topbar"><div className="topbar-left"><button className="icon-button mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={18} /></button><div className="breadcrumb">Knowledge base <span>/</span> <strong>New query</strong></div></div><div className="topbar-right"><span className={`service-state ${connected ? '' : 'state-offline'}`}><i />{connected ? 'Systems operational' : 'Service unavailable'}</span><span className="topbar-divider"/><button className="icon-button top-help" onClick={() => setToast('Ask a question to search your indexed sources.')} aria-label="Help"><CircleHelp size={16}/></button></div></header>
      <section className={`conversation ${messages.length ? 'conversation-active' : ''}`} ref={scrollRef}>
        {messages.length === 0 ? <div className="welcome-wrap"><div className="welcome-eyebrow"><span className="eyebrow-line"/>PRIVATE KNOWLEDGE, INSTANTLY<span className="eyebrow-line"/></div><div className="welcome-icon"><Sparkles size={19}/></div><h1>Good afternoon, Kavin<span>.</span></h1><p className="welcome-subtitle">Your knowledge base is ready.</p><p className="welcome-body">Ask a question across your connected sources. Every response includes citations and the context used to generate it.</p><div className="suggestion-grid">{suggestions.map(({icon: Icon,label,question: prompt})=><button key={label} className="suggestion-card" onClick={()=>void ask(prompt)}><span className="suggestion-icon"><Icon size={15}/></span><span><strong>{label}</strong><small>{prompt}</small></span><ArrowDown className="suggestion-arrow" size={13}/></button>)}</div><div className="grounding-note"><ShieldCheck size={14}/><span>{selectedDocs.length} sources included in retrieval</span><span className="note-divider"/><span>Answers grounded in source material</span></div></div> : <div className="message-list">{messages.map((message)=><article className={`message message-${message.role}`} key={message.id}><div className={`message-avatar ${message.role==='assistant'?'assistant-avatar':''}`}>{message.role==='assistant'?<Layers2 size={14}/>:<span>KP</span>}</div><div className="message-content"><div className="message-meta"><strong>{message.role==='assistant'?'Atlas':'You'}</strong><span>{message.pending?'Searching sources':message.role==='assistant'?'Response':'Just now'}</span>{message.role==='assistant' && !message.pending && !message.error && <button className="copy-button" onClick={()=>void copyLatest()} title={`Copy answer (${shortKey} ⇧ C)`}><Copy size={13}/><span>{copied?'Copied':'Copy'}</span></button>}</div>{message.pending?<div className="thinking"><span/><span/><span/><span>Retrieving relevant context…</span></div>:<div className={`markdown-body ${message.error?'error-copy':''}`}><ReactMarkdown components={{a:({href,children})=>href?.startsWith('citation:')?<button className={`inline-citation ${highlightedSource===href.slice(9)?'citation-active':''}`} onMouseEnter={()=>setHighlightedSource(href.slice(9))} onMouseLeave={()=>setHighlightedSource('')} onClick={()=>setDrawerSource(message.sources?.find(source=>source.source_id===href.slice(9))??null)}>{children}</button>:<a href={href}>{children}</a>}}>{renderedAnswer(message)}</ReactMarkdown></div>}{message.sources?.length ? <><button className="trace-toggle" onClick={()=>setTraceOpen(traceOpen===message.id?null:message.id)}><span className="trace-icon"><Activity size={13}/></span><span>Thinking &amp; retrieval process</span><span className="trace-count">{message.sources.length} passages</span><ChevronDown className={traceOpen===message.id?'chevron-up':''} size={14}/></button>{traceOpen===message.id&&<div className="trace-panel"><div className="trace-stats"><div><span>SEARCH MODE</span><strong>{message.retrieval?.mode??`${modeName} search`}</strong></div><div><span>PASSAGES SCANNED</span><strong>{message.retrieval?.candidates??'—'}</strong></div><div><span>RANKING METHOD</span><strong>{message.retrieval?.reranking?.split(';')[0]??'Hybrid rank fusion'}</strong></div></div><div className="trace-method">{message.retrieval?.reranking??'Dense vector + BM25 keyword ranking'}</div><div className="context-heading"><strong>Retrieved context</strong><span>{message.sources.length} chunks used</span></div>{message.sources.map((source,index)=><button className={`context-chunk ${highlightedSource===source.source_id?'chunk-hovered':''}`} key={source.source_id} onMouseEnter={()=>setHighlightedSource(source.source_id)} onMouseLeave={()=>setHighlightedSource('')} onClick={()=>setDrawerSource(source)}><div className="chunk-meta"><span className="citation-label">{source.source_id}</span><strong>{formatName(source.document)}</strong><span>§{source.section} · p. {source.page_start}{source.page_end!==source.page_start?`–${source.page_end}`:''}</span><span className="chunk-score">{message.retrieval?.mode?.startsWith('Sparse')?`BM25 ${source.lexical_score??0}`:source.similarity!==null&&source.similarity!==undefined?`${(source.similarity*100).toFixed(1)}% vector`:source.lexical_score?`BM25 ${source.lexical_score}`:'Keyword match'}</span></div><div className="chunk-title">{source.title}</div><p>{highlightText(source.text, messages[messages.findIndex((item)=>item.id===message.id)-1]?.content??'')}</p></button>)}</div>}<div className="sources"><div className="sources-label"><Check size={12}/> SOURCES</div><div className="source-list">{message.sources.map((source)=><button className="source-chip" key={source.source_id} onClick={()=>setDrawerSource(source)} onMouseEnter={()=>setHighlightedSource(source.source_id)} onMouseLeave={()=>setHighlightedSource('')}><span className="source-ref">{source.source_id}</span><span>{formatName(source.document)}</span><i>·</i><span>§{source.section}</span></button>)}</div></div></>:null}</div></article>)}</div>}
      </section>
      <div className="composer-region"><div className="composer-wrap">{error&&<div className="error-banner"><span>{error}</span><button onClick={()=>setError('')} aria-label="Dismiss error"><X size={14}/></button></div>}{toast&&<div className="toast-banner"><Check size={14}/><span>{toast}</span><button onClick={()=>setToast('')} aria-label="Dismiss notification"><X size={13}/></button></div>}<form className="composer" onSubmit={onSubmit}><textarea ref={textareaRef} value={question} onChange={(event)=>setQuestion(event.target.value)} onKeyDown={onKeyDown} placeholder="Ask anything across your knowledge base…" rows={1} aria-label="Ask a question"/><div className="composer-tools"><div className="composer-hints"><button type="button" className="attach-button" onClick={()=>fileInput.current?.click()} aria-label="Add a PDF"><Plus size={15}/></button><span className="composer-separator"/><button type="button" className="model-pill" onClick={()=>setToast('Model configured by your workspace administrator.')}>Gemini Flash <ChevronDown size={12}/></button><span className="composer-separator"/><div className="mode-switch" aria-label="Retrieval mode">{(['hybrid','dense','sparse'] as SearchMode[]).map((option)=><button type="button" key={option} className={mode===option?'mode-active':''} onClick={()=>setMode(option)}>{option==='hybrid'?'Hybrid':option==='dense'?'Dense':'Sparse'}</button>)}</div></div><div className="composer-submit"><span className="keyboard-hint"><kbd>↵</kbd> to send · <kbd>{shortKey} ↵</kbd> deep retrieve</span><button className="send-button" type="button" disabled={!question.trim()||busy} onClick={()=>void ask(question,true)} title={`Deep retrieve (${shortKey} + Enter)`}>{busy?<LoaderCircle className="spin" size={15}/>:<Zap size={15}/>}</button><button className="send-button send-quick" type="submit" disabled={!question.trim()||busy} aria-label="Send message" title="Send">{busy?<LoaderCircle className="spin" size={15}/>:<ArrowUp size={16}/>}</button></div></div></form><div className="composer-bottom"><span><ShieldCheck size={12}/> {selectedDocs.length} of {documents.length} sources active</span><span>AI answers can be incomplete. Verify critical information against the original source.</span></div></div><footer className="footer"><span>ATLAS <i>·</i> ENTERPRISE KNOWLEDGE</span><span><span className="footer-dot"/> PRIVATE WORKSPACE</span></footer></div>
    </main>
    {drawerSource&&<><button className="drawer-scrim" onClick={()=>setDrawerSource(null)} aria-label="Close source details"/><aside className="source-drawer"><div className="drawer-top"><div><span className="drawer-kicker">SOURCE REFERENCE</span><h2>Retrieved passage</h2></div><button className="icon-button" onClick={()=>setDrawerSource(null)} aria-label="Close"><X size={16}/></button></div><div className="drawer-document"><div className="pdf-icon"><FileText size={15}/></div><div><strong>{formatName(drawerSource.document)}</strong><span>Section {drawerSource.section} · {drawerSource.title}</span></div></div><div className="drawer-badges"><span>{drawerSource.source_id}</span><span>Page {drawerSource.page_start}{drawerSource.page_end!==drawerSource.page_start?`–${drawerSource.page_end}`:''}</span>{drawerSource.similarity!==null&&drawerSource.similarity!==undefined&&<span>{(drawerSource.similarity*100).toFixed(1)}% vector similarity</span>}</div><div className="drawer-content-label">EXACT RETRIEVED CONTEXT</div><div className="drawer-quote">{drawerSource.text}</div><button className="copy-citation" onClick={()=>{void navigator.clipboard.writeText(`[${drawerSource.source_id}] ${drawerSource.document}, Section ${drawerSource.section}, page ${drawerSource.page_start}`);setToast('Copied citation link');window.setTimeout(()=>setToast(''),2600)}}><Copy size={14}/> Copy citation link <kbd>{shortKey} ⇧ C</kbd></button><div className="drawer-note"><ShieldCheck size={14}/><span>This passage was included as grounding context for the answer.</span></div></aside></>}
    {paletteOpen&&<div className="palette-overlay" onMouseDown={(event)=>{if(event.target===event.currentTarget)setPaletteOpen(false)}}><div className="command-palette"><div className="palette-search"><Search size={17}/><input autoFocus placeholder="Search sources or run a command…" onChange={(event)=>setSearch(event.target.value)}/><kbd>ESC</kbd></div><div className="palette-section">QUICK ACTIONS</div><button onClick={()=>{setPaletteOpen(false);setMessages([]);textareaRef.current?.focus()}}><MessageSquareText size={15}/>New conversation<span>↵</span></button><button onClick={()=>{setPaletteOpen(false);fileInput.current?.click()}}><Plus size={15}/>Add knowledge source<span>⇧ A</span></button><button onClick={()=>{setPaletteOpen(false);void copyLatest()}}><Copy size={15}/>Copy latest answer<span>{shortKey} ⇧ C</span></button><div className="palette-section">SOURCES</div>{documents.filter(doc=>formatName(doc.filename).toLowerCase().includes(search.toLowerCase())).slice(0,5).map(doc=><button key={doc.filename} onClick={()=>{toggleDocument(doc.filename);setPaletteOpen(false)}}><FileText size={15}/>{formatName(doc.filename)}<span>{selectedDocs.includes(doc.filename)?'Included':'Excluded'}</span></button>)}</div></div>}
    <input ref={fileInput} className="hidden-input" type="file" accept="application/pdf,.pdf" onChange={(event)=>void onFile(event.target.files?.[0])}/>
  </div>
}
export default App
