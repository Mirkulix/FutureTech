import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "Z.ai Code Scaffold - AI-Powered Development",
  description: "Modern Next.js scaffold optimized for AI-powered development with Z.ai. Built with TypeScript, Tailwind CSS, and shadcn/ui.",
  keywords: ["Z.ai", "Next.js", "TypeScript", "Tailwind CSS", "shadcn/ui", "AI development", "React"],
  authors: [{ name: "Z.ai Team" }],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "Z.ai Code Scaffold",
    description: "AI-powered development with modern React stack",
    url: "https://chat.z.ai",
    siteName: "Z.ai",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Z.ai Code Scaffold",
    description: "AI-powered development with modern React stack",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className="antialiased bg-background text-foreground"
      >
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
            <div className="container mx-auto px-4 py-3 flex items-center justify-between gap-4">
              <Link href="/landing" className="flex items-center gap-2">
                <span className="h-7 w-7 rounded-lg bg-gradient-to-tr from-cyan-500 to-violet-500" />
                <div className="flex flex-col leading-tight">
                  <span className="text-sm font-semibold">FutureTech</span>
                  <span className="text-[11px] text-slate-400">AI Creation Platform</span>
                </div>
              </Link>
              <nav className="flex items-center gap-2 text-xs sm:text-sm">
                <Link href="/landing">
                  <Button variant="ghost" size="sm" className="text-slate-300">
                    Landing
                  </Button>
                </Link>
                <Link href="/">
                  <Button variant="ghost" size="sm" className="text-slate-300">
                    Training
                  </Button>
                </Link>
                <Link href="/live">
                  <Button variant="ghost" size="sm" className="text-slate-300">
                    Live
                  </Button>
                </Link>
                <Link href="/dashboard">
                  <Button variant="ghost" size="sm" className="text-slate-300">
                    Dashboard
                  </Button>
                </Link>
              </nav>
            </div>
          </header>
          <main className="flex-1">
            {children}
          </main>
          <Toaster />
        </div>
      </body>
    </html>
  );
}
