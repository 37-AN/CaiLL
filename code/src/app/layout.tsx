import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CaiLL AI Trading System - RL-Powered Trading Platform",
  description: "Advanced AI trading system powered by reinforcement learning. Monitor multi-agent RL training, track portfolio performance, and deploy intelligent trading strategies.",
  keywords: ["AI Trading", "Reinforcement Learning", "Trading Bot", "PPO", "DQN", "A2C", "Algorithmic Trading", "Machine Learning"],
  authors: [{ name: "CaiLL Team" }],
  icons: {
    icon: "/favicon.ico",
  },
  openGraph: {
    title: "CaiLL AI Trading System",
    description: "RL-powered algorithmic trading with multi-agent intelligence",
    siteName: "CaiLL",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "CaiLL AI Trading System",
    description: "RL-powered algorithmic trading platform",
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
