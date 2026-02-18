 import { Button } from '@/components/ui/button';
 import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
 import { Badge } from '@/components/ui/badge';
 import { Separator } from '@/components/ui/separator';
 import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
 import { Activity, Brain, Rocket, Zap, Layers, Shield, Wand2, Sparkles, BarChart3 } from 'lucide-react';
 import Link from 'next/link';
 
 export default function LandingPage() {
   return (
     <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
       <section className="relative overflow-hidden">
         <div className="absolute inset-0 opacity-40 blur-3xl">
           <div className="bg-gradient-to-tr from-indigo-600/30 via-violet-600/20 to-fuchsia-600/30 w-[60%] h-[60%] rounded-full -top-24 -left-24 absolute" />
           <div className="bg-gradient-to-tr from-cyan-600/30 via-teal-600/20 to-emerald-600/30 w-[50%] h-[50%] rounded-full bottom-0 right-0 absolute" />
         </div>
         <div className="container mx-auto px-6 py-16 relative">
           <div className="flex items-center gap-3 mb-8">
             <div className="p-3 rounded-lg bg-gradient-to-r from-blue-500 to-violet-500 shadow-lg shadow-blue-500/20">
               <Brain className="w-6 h-6" />
             </div>
             <div>
               <div className="text-sm text-slate-400">AI Creation SaaS</div>
               <h1 className="text-3xl font-bold tracking-tight">FutureTech</h1>
             </div>
             <Badge variant="outline" className="ml-auto bg-slate-800/60 border-slate-700">
               <Sparkles className="w-3 h-3 mr-1 text-yellow-400" />
               Beta Access
             </Badge>
           </div>
 
           <div className="grid lg:grid-cols-2 gap-10 items-center">
             <div>
               <h2 className="text-5xl font-extrabold leading-tight">
                 Erzeuge, trainiere und deploye
                 <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-violet-400"> eigene KI‑Modelle</span>
               </h2>
               <p className="mt-6 text-slate-300 text-lg">
                 Ein durchdachter End‑to‑End Flow: Datensets auswählen, TernaryLLM trainieren, Live‑Metriken verfolgen und in produktionsfähige Formate exportieren – alles in Minuten.
               </p>
               <div className="mt-8 flex flex-wrap gap-3">
                 <Link href="/">
                   <Button size="lg" className="bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-lg shadow-blue-500/30">
                     <Rocket className="w-4 h-4 mr-2" />
                     Zum Trainings‑Cockpit
                   </Button>
                 </Link>
                 <Link href="/dashboard">
                   <Button size="lg" variant="outline" className="border-slate-700 bg-slate-900/40">
                     <BarChart3 className="w-4 h-4 mr-2" />
                     Monitoring & Exports
                   </Button>
                 </Link>
               </div>
               <div className="mt-6 flex items-center gap-4">
                 <Badge variant="outline" className="bg-slate-800/50">Serverless‑Ready</Badge>
                 <Badge variant="outline" className="bg-slate-800/50">GGUF & HF‑Export</Badge>
                 <Badge variant="outline" className="bg-slate-800/50">Live‑Metriken</Badge>
               </div>
             </div>
 
             <Card className="bg-slate-900/60 border-slate-800 backdrop-blur-sm">
               <CardHeader>
                 <CardTitle className="flex items-center gap-2">
                   <Activity className="w-5 h-5 text-emerald-400" />
                   Live Training Preview
                 </CardTitle>
                 <CardDescription>Ein Blick auf die Performance während des Trainings</CardDescription>
               </CardHeader>
               <CardContent>
                 <div className="grid grid-cols-3 gap-4">
                   <Card className="bg-slate-950/60 border-slate-800">
                     <CardHeader className="pb-2">
                       <CardTitle className="text-sm">Tokens/sec</CardTitle>
                     </CardHeader>
                     <CardContent>
                       <div className="text-2xl font-bold">4.8k</div>
                       <div className="text-xs text-slate-400">durchschnittlich</div>
                     </CardContent>
                   </Card>
                   <Card className="bg-slate-950/60 border-slate-800">
                     <CardHeader className="pb-2">
                       <CardTitle className="text-sm">Aktueller Loss</CardTitle>
                     </CardHeader>
                     <CardContent>
                       <div className="text-2xl font-bold">1.23</div>
                       <div className="text-xs text-slate-400">sinkend</div>
                     </CardContent>
                   </Card>
                   <Card className="bg-slate-950/60 border-slate-800">
                     <CardHeader className="pb-2">
                       <CardTitle className="text-sm">Sparsity</CardTitle>
                     </CardHeader>
                     <CardContent>
                       <div className="text-2xl font-bold">78%</div>
                       <div className="text-xs text-slate-400">Ternary aktiv</div>
                     </CardContent>
                   </Card>
                 </div>
                 <Separator className="my-6 bg-slate-800" />
                 <Tabs defaultValue="features" className="w-full">
                   <TabsList className="bg-slate-800/50 border border-slate-700">
                     <TabsTrigger value="features">Features</TabsTrigger>
                     <TabsTrigger value="workflow">Workflow</TabsTrigger>
                     <TabsTrigger value="export">Export</TabsTrigger>
                   </TabsList>
                   <TabsContent value="features" className="mt-4">
                     <div className="grid grid-cols-2 gap-4">
                       <Card className="bg-slate-950/60 border-slate-800">
                         <CardHeader className="pb-2">
                           <CardTitle className="text-sm flex items-center gap-2"><Wand2 className="w-4 h-4 text-fuchsia-400" /> Model Presets</CardTitle>
                         </CardHeader>
                         <CardContent className="text-sm text-slate-300">Tiny, Small, Base, Large – in 1 Klick konfigurierbar.</CardContent>
                       </Card>
                       <Card className="bg-slate-950/60 border-slate-800">
                         <CardHeader className="pb-2">
                           <CardTitle className="text-sm flex items-center gap-2"><Layers className="w-4 h-4 text-cyan-400" /> Curriculum & Filtering</CardTitle>
                         </CardHeader>
                         <CardContent className="text-sm text-slate-300">Entropy‑basiertes Filtering und Curriculum Learning.</CardContent>
                       </Card>
                       <Card className="bg-slate-950/60 border-slate-800">
                         <CardHeader className="pb-2">
                           <CardTitle className="text-sm flex items-center gap-2"><Zap className="w-4 h-4 text-yellow-400" /> Ternary Optimierungen</CardTitle>
                         </CardHeader>
                         <CardContent className="text-sm text-slate-300">Hohe Sparsity und Speicher‑Effizienz.</CardContent>
                       </Card>
                       <Card className="bg-slate-950/60 border-slate-800">
                         <CardHeader className="pb-2">
                           <CardTitle className="text-sm flex items-center gap-2"><Shield className="w-4 h-4 text-emerald-400" /> Export‑Sicherheit</CardTitle>
                         </CardHeader>
                         <CardContent className="text-sm text-slate-300">GGUF/HF‑Exports mit überprüfter Integrität.</CardContent>
                       </Card>
                     </div>
                   </TabsContent>
                   <TabsContent value="workflow" className="mt-4">
                     <div className="text-sm text-slate-300">
                       Dataset wählen → Modell konfigurieren → Training starten → Live überwachen → Exportieren.
                     </div>
                   </TabsContent>
                   <TabsContent value="export" className="mt-4">
                     <div className="text-sm text-slate-300">
                       GGUF für llama.cpp, HuggingFace‑Format oder rohe Gewichte. Alles direkt aus dem Dashboard.
                     </div>
                   </TabsContent>
                 </Tabs>
               </CardContent>
             </Card>
           </div>
         </div>
       </section>
 
       <section className="container mx-auto px-6 py-16">
         <h3 className="text-2xl font-bold mb-6">Warum FutureTech</h3>
         <div className="grid md:grid-cols-3 gap-6">
           <Card className="bg-slate-900/60 border-slate-800">
             <CardHeader>
               <CardTitle className="flex items-center gap-2"><Rocket className="w-5 h-5 text-violet-400" /> Schnell vom Konzept zum Modell</CardTitle>
               <CardDescription>Ein klarer Flow, keine komplizierte Infrastruktur.</CardDescription>
             </CardHeader>
           </Card>
           <Card className="bg-slate-900/60 border-slate-800">
             <CardHeader>
               <CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-emerald-400" /> Live‑Metriken & Replays</CardTitle>
               <CardDescription>Training sichtbar machen, Runs wiedergeben, vergleichen.</CardDescription>
             </CardHeader>
           </Card>
           <Card className="bg-slate-900/60 border-slate-800">
             <CardHeader>
               <CardTitle className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-yellow-400" /> Export in produktionsfähige Formate</CardTitle>
               <CardDescription>GGUF & HF für nahtlose Integration in deine Umgebung.</CardDescription>
             </CardHeader>
           </Card>
         </div>
         <div className="mt-10 flex gap-4">
           <Link href="/">
             <Button size="lg" className="bg-gradient-to-r from-cyan-600 to-blue-600">Jetzt trainieren</Button>
           </Link>
           <Link href="/dashboard">
             <Button size="lg" variant="outline" className="border-slate-700 bg-slate-900/40">Monitoring öffnen</Button>
           </Link>
         </div>
       </section>
 
       <footer className="border-t border-slate-800 bg-slate-900/40">
         <div className="container mx-auto px-6 py-10 text-slate-400 text-sm">
           © {new Date().getFullYear()} FutureTech. Alle Rechte vorbehalten.
         </div>
       </footer>
     </main>
   )
 }
