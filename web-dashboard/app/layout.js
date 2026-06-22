import "./globals.css";
import { AuthProvider } from "./auth-context";

export const metadata = {
  title: "DriveFlow",
  description: "Web-based gesture dashboard ready for Vercel deployment",
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
