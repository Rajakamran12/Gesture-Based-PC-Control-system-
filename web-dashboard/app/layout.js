import "./globals.css";
import { AuthProvider } from "./auth-context";

export const metadata = {
  title: "AI-Driven Gesture Based PC Control System",
  description: "AI-Driven Gesture Based PC Control System",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
